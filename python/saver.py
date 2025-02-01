import pathlib, torch, labml
from labml.experiment import ModelSaver

class TorchSaver(ModelSaver):
    def __init__(self, name: str, model, to_load = True):
        self.name = name
        self.model = model
        self.to_load = to_load

    def save(self, checkpoint_path: pathlib.Path) -> any:
        state = self.model.state_dict()
        file_name = f"{self.name}.pth"
        torch.save(state, str(checkpoint_path / file_name))
        return file_name

    def load(self, checkpoint_path: pathlib.Path, info: any):
        if not self.to_load: return
        file_name: str = info
        self_state = self.model.state_dict()
        try:
            sample_param = next(iter(self_state.values()))
            device = sample_param.device
        except AttributeError:
            device = torch.device('cpu')
        except StopIteration:
            device = torch.device('cpu')

        loaded_state = torch.load(str(checkpoint_path / file_name), weights_only=True, map_location=device)

        if self.name != 'model':
            self.model.load_state_dict(loaded_state)
            return

        if set(loaded_state) != set(self_state):
            raise RuntimeError('Model not compatible')
        if all([loaded_state[i].shape == self_state[i].shape for i in loaded_state]):
            self.model.load_state_dict(loaded_state)
            return

        # expand model without affecting output
        for i in loaded_state:
            si, li = self_state[i], loaded_state[i]
            if si.shape == li.shape:
                self_state[i] = li
            elif len(si.shape) == 4: # Conv2d: (outch, inch, kx, ky)
                if si.shape[2:] != li.shape[2:]:
                    raise RuntimeError('Kernel size not compatible')
                if any([si.shape[i] < li.shape[i] for i in range(2)]):
                    raise RuntimeError('Cannot shrink model')
                self_state[i][:li.shape[0],:li.shape[1]] = li
                self_state[i][:li.shape[0],li.shape[1]:] = 0
            elif len(si.shape) == 3: # LayerNorm: (ch, h, w)
                if si.shape[1:] != li.shape[1:]:
                    raise RuntimeError('Feature size not compatible')
                if si.shape[0] < li.shape[0]:
                    raise RuntimeError('Cannot shrink model')
                self_state[i][:li.shape[0]] = li
            elif len(si.shape) == 2: # Linear: (out, in)
                if any([si.shape[i] < li.shape[i] for i in range(2)]):
                    raise RuntimeError('Cannot shrink model')
                self_state[i][:li.shape[0],:li.shape[1]] = li
                self_state[i][:li.shape[0],li.shape[1]:] = 0
            elif len(si.shape) == 1: # BatchNorm: (ch,) / bias for Conv2d/Linear/BatchNorm
                self_state[i][:li.shape[0]] = li
            else:
                raise RuntimeError('Model not compatible')
        self.model.load_state_dict(self_state)
