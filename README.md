# BetaTetris Tablebase

<p align="center" width="100%">
<img src="https://raw.githubusercontent.com/adrien1018/betatetris-tablebase/main/icon.png" alt="" width="200"/>
</p>

You can now try the neural network of BetaTetris on [BetaTetris Playground](https://fractal161.github.io/btpg/)!

BetaTetris Tablebase is an NES Tetris agent incorporating a tablebase to brute-force optimal moves on a predefined set of boards. It also contains a hybrid agent that merges the tablebase approach with the previously established [neural-net-based BetaTetris](https://github.com/adrien1018/beta-tetris).

This is the BetaTetris version featured in [Tetris Friendlies Revolution 2023](https://www.youtube.com/watch?v=ZeCa5FYaJ3I), and is also the version that achieves [the first AI rebirth](https://www.youtube.com/watch?v=yBOfsnBQ5i0).

## Table of Contents

- [Introduction](#introduction)
- [Statistics](#statistics)
- [FAQ](#faq)
- [Running / generating the agent](#running--generating-the-agent)
    - [NN](#nn)
    - [Tablebase](#tablebase)
- [Contact me](#contact-me)
- [License](#license)

## Introduction

The basic concept of BetaTetris Tablebase is as follows. Consider a Tetris variant: we first collect a set of boards and play Tetris normally, except that you are limited to play on those boards.
This variant has a useful property: it is possible to create a bot that plays *perfectly* in this variant, in the sense that it achieves the highest possible average score. This is accomplished through an extensive tree search (expectimax) that calculates the highest possible average score for any given board, line count, and current piece, which are then stored in a tablebase.
Although this version differs from the original Tetris game, by using enough boards in the set of boards, the perfect bot on this variant can also play decently well on normal Tetris.

The agent featured in Tetris Friendlies Revolution 2023 used approximately 3.5 billion boards gathered in millions of games played by previous BetaTetris versions. With such a large set of boards, the tablebase outperforms any previous BetaTetris versions in pre-29 play. However, it falls short after killscreen where digs are more likely to happen.
Consequently, a hybrid approach is used: before level 37, if the expected score calculated by the tablebase exceeds a certain threshold, the tablebase decides the placement; otherwise, the decision is made by the neural network. After level 37, the neural network has the full control.

This agent is capable of playing at any tap speed, reaction time, and line cap. For additional insights, you may view the [introduction video](https://youtu.be/ZeCa5FYaJ3I?t=1038) from Tetris Friendlies Revolution 2023.

## Statistics

### Newest NN model (v1.0.0)

This is a newly-trained model which supports multiple tap speeds, reaction times and aggression levels, and significantly outperforms the previous version. See the [statistics page](https://betatetris.github.io/btpg/stats/) in [btpg](https://betatetris.github.io/btpg/) for its statistics.

### Old NN model & tablebase (v0.1.0)

This is the agent used in [Tetris Friendlies Revolution 2023](https://www.youtube.com/watch?v=ZeCa5FYaJ3I), and the version that achieves [the first AI rebirth](https://www.youtube.com/watch?v=yBOfsnBQ5i0). It uses a smaller NN model which only supports 30 Hz, 300 ms reaction time. A 3.5-billion-board tablebase is used.

These statistics were collected through simulations of the agents playing on 2000 (tablebase, NN) or 1100 (hybrid) randomly-selected TetrisGYMv5 seeds. The format is 30 Hz, 300 ms reaction time, level 39 double killscreen and level 49 line cap.

|       |                 | Tablebase | NN        | Hybrid    |
|-------|-----------------|----------:|----------:|----------:|
| Lines | Average         | 369.4     | 392.1     | 399.9     |
|       | Std. dev.       | 66.1      | 60.4      | 36.2      |
|       | Median          | 385       | 419       | 419       |
|       | <130 (%)        | 1.0       | 1.3       | <0.1      |
|       | <230 (%)        | 2.6       | 3.2       | <0.1      |
|       | <330 (%)        | 19.8      | 7.1       | 2.0       |
| Score | Average         | 2,170,127 | 2,342,345 | 2,426,105 |
|       | Std. dev.       |   410,438 |   423,790 |   270,280 |
|       | 1st percentile  |   615,600 |   484,120 | 1,788,340 |
|       | 5th percentile  | 1,392,200 | 1,664,180 | 1,992,180 |
|       | 10th percentile | 1,571,400 | 1,947,620 | 2,057,200 |
|       | 30th percentile | 2,081,420 | 2,236,000 | 2,261,400 |
|       | 50th percentile | 2,244,100 | 2,435,780 | 2,459,060 |
|       | 70th percentile | 2,410,020 | 2,573,140 | 2,606,260 |
|       | 90th percentile | 2,577,960 | 2,727,400 | 2,745,740 |
|       | 95th percentile | 2,644,460 | 2,786,820 | 2,818,800 |
|       | 99th percentile | 2,730,400 | 2,935,780 | 2,919,720 |
|       | pre-19 avg.     |   655,674 |   636,847 |   657,892 |
|       | pre-29 avg.     | 1,280,030 | 1,228,906 | 1,293,916 |
|       | pre-39 avg.     | 1,946,062 | 1,962,601 | 2,038,563 |

The "<0.1" is due to the hybrid agent got at least 230 lines in all of the 1100 games it played.

## FAQ

Q: Does the tablebase play **perfectly** on Tetris?\
A: **No!** The tablebase only plays perfectly in a hypothetical variant of Tetris, where you're restricted to playing only on the boards included in the tablebase, which we'll call "*Tetris Limited*" from now on.

Q: If that's the case, does the tablebase play **perfectly** on Tetris provided it never encounters boards outside the tablebase?\
A: Again, **no!** There's a subtle but important difference between "Tetris, but you can only play on the boards used in the tablebase" and "Tetris, but you happened to be able to play on only the boards used in the tablebase", though it may seem confusing at the first glance.
The distinction lies in **expectation**. In the first scenario (where the tablebase is perfect on), you are aware of the fact that playing to non-tablebase boards means an instant game over, so you can adjust your strategy accordingly. In the second scenario, you're playing regular Tetris with a normal strategy; it's just that the RNG aligns in a way that allows you to keep playing within the tablebase boards.\
And as a result, we can also expect the tablebase will play more safely (less aggressive / efficient) than perfect play on normal Tetris, because it will view boards outside the tablebase as an immediate top-out and try to avoid them at all costs, even if some of them are survivable.\
If the difference is still confusing to you, consider the optimal strategies in normal Tetris versus drought mode. It should be clear that the optimal play on drought mode should be a lot safer than the optimal play on normal Tetris because you are **expecting** much longer droughts, even if it's theoretically possible (albeit highly unlikely) to encounter the exact same piece sequence in both modes.

Q: Does the tablebase play **perfectly** on *Tetris Limited*?\
A: Yes. In *Tetris Limited*, the tablebase provably achieves the **best possible average score**, so it is perfect in this regard.
However, "perfect" can be defined in various ways, for example, best possible 99-th percentile score (for aggression), best possible pre-killscreen score, etc.
The tablebase doesn't meet these definitions of perfection, and attaining such perfection in these aspects might be extremely challenging, even in *Tetris Limited*.

Q: How do you prove the tablebase achieves the best possible average score (on *Tetris Limited*)?\
A: The tablebase uses expectimax algorithm (not to be confused with expectiminimax, which is used in adversarial games), which selects moves that maximize the average score at each decision point.
By the [linearity of expectation](https://en.wikipedia.org/wiki/Expected_value#Properties), the optimality of every individual move implies the optimality of the overall score.

Q: How well does BetaTetris perform compared to other AIs? Is it currently the best AI?\
A: As of December 2023, BetaTetris is likely the strongest bot in terms of average score, in the format of level 39 double killscreen, level 49 cap, 30 Hz tapping and 300ms reaction time.
However, StackRabbit is more aggressive than BetaTetris in pre-29 play, and may outperform BetaTetris in most of the games in this phase, though I'm cannot say for sure due to a lack of statistics.\
Generally, determining a single "best" bot is complex - just like when defining "perfect", there are too many metrics to compare bots.
Bots could even fall into a rock-paper-scissors situation where one bot's strategy defeats another's (winning more than half of the games) in a circular way. The only way to objectively compare AI's strengths and weaknesses is to compare their distributions of scores and lines.

Q: Can BetaTetris (NN or tablebase) do RNG manipulation?\
A: No. Both the NN and the tablebase assume a Markov-style RNG (which means the distribution of the next piece only depends on the current piece). This differs from the RNG in real NES Tetris, though they are quite similar.
The distinction prevents RNG manipulation, but the similarity still allows the bots to perform well on the original RNG.\
(This also implies that the tablebase is only perfect **on the Markov-style RNG** on *Tetris Limited* - but well, the perfect strategy on the real RNG will just be RNG manipulation🙂)

Q: How did you collect the boards used in the tablebase?\
A: I run millions of games using old BetaTetris versions and collected all the encountered boards. No filtering was done. The games are in several different formats with tap speed ranging from 12 to 30 Hz and reaction time ranging from 8 to 21 frames.

Q: What are the respective strengths and weaknesses of NN and tablebase, and why is the hybrid approach so effective?\
A: The tablebase's strength lies in its optimality. Its weaknesses include its inability to play on dirty boards (because there are too many dirty boards compared to clean boards) and a tendency to play overly safe due to assuming instant top-out for unseen boards (discussed in the second question).\
The strength of NN is its ability to generalize, recognizing board patterns rather than memorizing them like the tablebase. Thus, it is able to deal with a lot of different situations such as digs. However, it may struggle in less-trained situations, leading to suboptimal decisions.\
The hybrid approach leverages the strengths of both: when the board is clean, the tablebase can unleash its full potential of exhaustive search, and the average score is also a very good indicator of the cleanness of the board.
Once the board start getting dirty, it will be reflected on the average score, and the NN can take over in time to utilize its pattern-recognizing ability to navigate the dig.

Q: Why do you disable the tablebase after level 37? Does this mean the tablebase cannot play on double killscreen? And why choose level 37 instead of 39?\
A: The tablebase can play on double killscreen. However, it will line out much more instead of going for tetrises, due to the fact that efficient double killscreen playing requires navigating boards with lots of holes / overhangs, which would require a substantially larger tablebase. Thus, I disabled the tablebase entirely on double killscreen.\
The decision to switch at level 37, rather than 39, aims to give the NN some time to adapt to a board that it is more comfortable with before the double killscreen transition. This choice is a heuristic and not experimentally verified; it is very possible that the optimal switching point is not level 37.

Q: Why does the introduction video show the tablebase's average score as around 2.158 million, while the statistics section indicates 2.17 million?\
A: This is due to a bug in the tablebase's score calculation when I generated the tablebase for the event. Specifically, I erronously implemented the score calculation to use the pre-clear level instead of the post-clear level in score computation, resulting in a slightly lower score. The bug is fixed now.\
(Fun fact: This bug was present in all previous BetaTetris versions, though its hardly affects anything so that I didn't find it until comparing the statistics of the tablebase!)\
Additionally, the RNG variance between Markov-style and real RNG might contribute to this difference, but the average score difference between the Markov-style RNG and the real RNG is well within the error margin in my experiments.

Q: How can the tablebase be improved?\
A: There are a lot of possibilities, but here are some most promising approaches:
- Use an NN (or whatever other algorithm) to estimate the average score of any unseen board, so that the tablebase will no longer assume an instant topout for unseen boards, solving its biggest weakness.
    - Anything that don't over-estimate the average score will be an improvement. Overestimating is relatively bad because the tablebase might find a way to play toward the overestimated boards.
- Refining the board set by filtering out bad boards and adding good boards.
    - Bad boards can be identified simply by looking at the average score in the tablebase, while good boards can be obtained from other agent's playouts or generated through simple evaluation functions from existing boards.

There are definitely many other potential improvements. Feel free to fork the repository and experiment with your modifications!

Q: Can I get your board set to run the agent / do some research or anything?\
A: Absolutely! The board set is quite large and I don't currently have a server to host them, but you can [contact me](#contact-me) and I will figure out a way to send the file to you.

## Running / generating the agent

There are two primary components to consider: Tablebase and NN. Below is a basic guide on how to use each.
The NN is relatively easy to run, whereas setting up the Tablebase is more complex, as it requires generating the tablebase from scratch (the tablebase files enabling the hybrid agent are approximately 1.5TB in size, and I am currently unable to host them).

You will need a machine equipped with BMI2 and AVX2 instruction sets. All Intel Haswell (4th generation) or newer CPUs, as well as AMD Ryzen series CPUs, should support them.

### NN

#### Run

You will need a machine with Python and a relatively new C++ compiler installed (recommended compiler: Visual Studio 2022 or newer on Windows, and GCC 12 or newer on Linux).
On Windows, it will also be very handy to install Git Bash for running commands.

It is highly recommended to have a CUDA-capable GPU available. Otherwise, the performance may suffer.

To begin, navigate to the `python/` directory and install the required Python packages:

```bash
pip install -r requirements.txt
```

Next, proceed to `python/tetris/` to build the C++ extension:
```bash
python setup.py build_ext --inplace
```

With these steps completed, the NN is ready for use! First, download the model from the [Release page](https://github.com/BetaTetris/betatetris-tablebase/releases). Then, return to `python/` and start the FCEUX server:
```bash
python fceux.py --tap-speed=30hz --adj-delay=21 --aggression=low [path-to-model]
```

Finally, open FCEUX and run the Lua script in the `lua/` directory to watch the NN in action!

#### Train

To train your own BetaTetris NN model (possibly for a different format), you'll need a CUDA-enabled GPU with at least 8GB of VRAM, preferably with good FP16 performance.

Firstly, modify the `extra_compile_args` in `python/tetris/setup.py` to set the tap speed, line cap, and reaction time (measured in frames). For a detailed format, see [Tablebase#Compile](#compile).

Next, rebuild the C++ extension in `python/tetris/`:
```bash
rm -rf build/
python setup.py build_ext --inplace
```

Then, start a LabML server, which provides a website interface for monitoring training progress and adjusting hyperparameters in real-time:
```bash
pip install labml-app==0.0.102
labml app-server --ip 127.0.0.1 --port 12345
```

Create a `.labml.yaml` file under `python/` to specify the model storage location and the LabML server address:
```yaml
data_path: [path to store the models]
web_api_frequency: 60
web_api: 'http://127.0.0.1:12345/api/v1/track?'
web_api_open_browser: false
```

Finally, navigate to `python/` and start the training by running `train.py`!
```bash
python train.py [run-name]
```

`[run-name]` serves as an identifier for the training session. To resume a previous training run after an interruption or modifying some code, use:
```bash
python train.py [run-name] [run-number]
```

Here, `[run-number]` is the number in the generated run ID, which can be found in the training output. For example, if the output displays:
```
run-name: run-name-005
[clean]: ""
Monitor experiment at http://127.0.0.1:12345/run/run-name-005
```
then use a `[run-number]` of 5 if you want to resume this run. You can also just input `last` if you want to resume the last run.

Command-line options are available to set various hyperparameters (like model size, learning rate, etc.). Run `python train.py --help` for a complete list of available hyperparameters.

Contributions by submitting of models for different formats are encouraged. Should you encounter any difficulties while training the models, please feel free to [contact me](#contact-me) for assistance.

### Tablebase

#### Compile

The tablebase is only tested in a Linux environment and requires CMake and GCC 12 (or newer) to be installed.

To compile the program, follow the standard procedure for CMake:
```bash
mkdir build && cd build
cmake -DTAP_SPEED=Tap30Hz -DADJ_DELAY=18 -DLINE_CAP=430 ..
make -j8
```

In the `cmake` command, you can specify the tap speed, line cap, and reaction time (measured in frames).
The tap speed options include `Tap12Hz`, `Tap15Hz`, `Tap20Hz`, `Tap30Hz`, or you can define an arbitrary tap speed using the format `'move_search::TapTable<1,2,3,2,3,2,3,2,3,2>'`.
In this format, the first number represents the delay of the initial tap, followed by nine numbers indicating the number of frames between subsequent taps.
For example, a frame pattern of `.X.X..X.X..X.X..X.X..X.X` corresponds to the numbers provided in the example above. The reason that there are 10 inputs is that it is possible to adjust a piece all the way from left to right and tuck it right after.

A executable file `main` should be generated after a successful compilation. This is the only program required for all operations related to the tablebase.

#### Generate the tablebase

To generate the tablebase, you first need a bunch of boards to start with. The system resources required can be estimated as follows:
- Memory: 24MB per million boards
- Disk: 250-450MB per million boards for strategy generation; more disk usage can speed up computation. 6GB per million boards for full tablebase (expected value + standard deviation) output. SSD is highly recommended.
- CPU: 8-12 cores should be fine for any number of boards since the speed is limited mainly by disk and memory access speed.
- Time: 7 minutes per million boards given a fast enough SSD. Note that fewer CPU cores or slower disks will result in longer processing times, while parallelizing some of the steps (needs more RAM, CPU and disk) can reduce the time.
In order to have a good enough tablebase (with reasonable Tetris rate), it's advisable to use a minimum of 100 million boards.

The set of boards should be stored in a file using the following format:
Each board is denoted by 200 bits, representing the 200 cells in a Tetris playfield. Here, an 1 indicates an empty cell and 0 a filled cell, arranged from top to bottom, left to right.
These 200 bits are compacted into 25 bytes, sequenced from the first to the last byte, with LSB to MSB order.
The board set should be stored in a file of N * 25 bytes, with each board concatenated sequentially. Ensure no duplicate boards are present, and the number of filled cells in every board is even.

After preparing the board file, execute the following commands to generate the boards and the edges (valid moves):
```bash
./main preprocess -p 5 [workdir] [board file]
./main build-edges -p 16 -g 0:5 [workdir]
```
- `[workdir]` serves as the storage location for all tablebase-related data. It should have sufficient disk space, ideally on a fast SSD.
- After running `preprocess`, the original board file can be discarded as it is now stored in the working directory in a format ready for further processing.
- If sufficient CPU cores or memory are available, multiple `build-edges` processes can be executed concurrently by assigning distinct `-g` values. Each of the values 0, 1, 2, 3, 4 must be used exactly once (in any order). For instance, one process can be run with `-g 0,1,2` while another with `-g 3,4`.
- `-p` denotes the level of parallelism. Reduce this value if fewer CPU threads are available. Using a parallelism setting higher than recommended may not yield significant performance improvements, or might even lead to slowdowns, unless your RAM and SSD are exceptionally fast.

After this, generate some checkpoints. This would simplify subsequent steps:
```bash
./main evaluate -p 6 -c 0:1081:40 [workdir] 2>&1 | tee evaluate.log
```

- `-c 0:1081:120` indicates the checkpoints to be stored. `0:1081:120` means storing a checkpoint every 120 pieces, up to 1080 pieces, similar to Python slicing syntax.
    - Average scores are calculated from the last to the first piece in the game. Only the average scores of the subsequent piece are needed to calculate the current piece's score. A checkpoint, therefore, is all the average scores at a certain piece count saved in a file.
- If the program is unexpectedly terminated, you can resume from a previously stored checkpoint using `-r [checkpoint]`.
- More checkpoints increase disk usage but allow greater parallelism in later steps.
- Alter checkpoints if a different line cap is used. The maximum number of pieces can be calculated as ((maximum filled cell in any board + line cap \* 10) / 4).
- The logs can be useful later so we store it by `tee`.

After generating the checkpoints, proceed to generate the placements:
```bash
./main move -p 4 -e 960 [workdir]
./main move-merge -p 5 -s 960 -e [max-piece] -d [workdir]
./main move -p 4 -r 960 -e 840 [workdir]
./main move-merge -p 5 -s 840 -e 960 -d [workdir]
./main move -p 4 -r 840 -e 720 [workdir]
./main move-merge -p 5 -s 720 -e 840 -d [workdir]
./main move -p 4 -r 720 -e 600 [workdir]
./main move-merge -p 5 -s 600 -e 720 -d [workdir]
...
./main move -p 4 -r 0 -e 120 [workdir]
./main move-merge -p 5 -s 0 -e 120 -d [workdir]

./main move-merge -p 5 -w -d [workdir] # final merge
```
- `[max-piece]` varies based on your board set and line cap. Use the value (1 + (piece number shown in the first line of output when running `move`)).
- The `move` commands generate placements for pieces between `-e` and `-r` parameters. The `move-merge` commands (except the final merge) convert placements into a more compressible format to save disk space.
- The `move` command requires the existence of the checkpoint specified by `-r`. The `move-merge` command requires computed placements. You can thus modify the exact ranges in the commands, the command order, or even run some commands in parallel, depending on the available checkpoints, disk space, and RAM.
    - For instance, if you have sufficient disk space & RAM, you can run all `move` commands in parallel, and then run `./main move-merge -p 5 -s 0 -e [max-piece] -d [workdir] && ./main move-merge -p 5 -w [workdir]` at the end.

At this point, the tablebase can already operate on its own and play some games! However, to use the hybrid agent, another step is required to generate the "confidence level" of any given board.
The confidence level of a board is defined as the ratio of the average score to a reference "threshold" value. The confidence levels used in the Tetris Friendlies Revolution can be generated with these commands:
```bash
./main threshold -p 4 -b 16 -l 0.5 -h 1 -e 960 -f [threshold_file] [workdir] [threshold_name]
./main threshold-merge -p 5 -s 960 -e [max-piece] -d [workdir] [threshold_name]
./main threshold -p 4 -b 16 -l 0.5 -h 1 -r 960 -e 840 -f [threshold_file] [workdir] [threshold_name]
./main threshold-merge -p 5 -s 840 -e 960 -d [workdir] [threshold_name]
./main threshold -p 4 -b 16 -l 0.5 -h 1 -r 840 -e 720 -f [threshold_file] [workdir] [threshold_name]
./main threshold-merge -p 5 -s 720 -e 840 -d [workdir] [threshold_name]
./main threshold -p 4 -b 16 -l 0.5 -h 1 -r 720 -e 600 -f [threshold_file] [workdir] [threshold_name]
./main threshold-merge -p 5 -s 600 -e 720 -d [workdir] [threshold_name]
...
./main threshold -p 4 -b 16 -l 0.5 -h 1 -r 120 -e 0 -f [threshold_file] [workdir] [threshold_name]
./main threshold-merge -p 5 -s 0 -e 120 -d [workdir] [threshold_name]

./main threshold-merge -p 5 -w -d [workdir] [threshold_name] # final merge
```

- The commands follow the same logic as the previous placement generation. The `[max-piece]`, `-e`, `-r`, `-s` parameters have the same meaning as previously described.
- `[threshold_file]` should be a file containing 430 (or whatever the line cap is) lines, each with a number. The i-th number (0-based) indicates the threshold value for boards with line count i. The values used in the Tetris Friendlies Revolution are the average scores of an empty board at each line count (interpolated at odd line counts), available in the `example_threshold` file. For different board sets, obtain the empty board's average score from the `val0` output at piece counts divisible by 5 when running `evaluate`.
- `-b` is the granularity of confidence levels; `-l`, `-h` are the ratios corresponding to the minimum and maximum confidence levels. For example, `-b 16 -l 0.5 -h 1` means all boards with average scores less than 0.5\*threshold will be assigned confidence level 0, and those greater than 1\*threshold will be assigned confidence level 15, with equal division for intermediate levels.
- `[threshold_name]` is an arbitrary name for the confidence level. You can use different threshold files and parameters to generate other confidence levels by specifying a different `[threshold_name]`.

#### Watch it in action

For pure tablebase play, simply start the FCEUX server:

```bash
./main fceux-server [workdir]
```

For hybrid agent (tablebase + NN), first start the board server and run `fceux.py` specifying the board server:

```bash
./main board-server -p 3457 [workdir] [threshold_name]
# in another terminal, under python/
python fceux.py -s localhost:3457 -p 3456 models/30hz-18f.pth
```

In both cases, open FCEUX and run the Lua script in `lua/` to watch the agent play!

## Contact me

If you encounter any issues related to BetaTetris, including difficulties in running, training, or generating the agents, feel free ping me on Discord (@adrien1018). You can contact me either through DM or in the #ai channel of the [CTM Discord channel](https://discord.gg/monthlytetris).

## License

BetaTetris Tablebase, including the neural network, is licensed under the terms of the GNU General Public License v3.
