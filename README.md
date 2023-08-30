
# Master Thesis: Lokale Navigation von Mikromobilitätsfahrzeugen mittels Reinforcement Learning

This repository outlines the results of my master thesis on training a little robot
in my own simulator and letting it interact with pedestrians controlled by Social Force.

The results show a very promising defensive policy that manages to avoid collisions entirely,
given a very dense environment like in crowded pedestrian zones. This defensive policy can be
combined with an offensive policy to maneuver around pedestrians on side-walks with only
a few pedestrians.

![](./results/videos/jetpack.gif)

## Main Results

- [thesis](./thesis.pdf)
- [presentation](./presentation.pdf)
- [demo video (defensive)](./demo/demo_defensive_policy.mkv)
- [demo video (offensive)](./demo/demo_offensive_policy.mkv)

## Structure

- code/ contains the code of robot-sf and pysocialforce, as well as the (unfinished) dreamer implementation
- maps/ contains the training environments that were used to train the robot
- proofs/ contains a mathematical proof of the obstacle force as a virtual potential field
- results/ contains videos of the policies, trained agents, training logs and performance profiles
- demo/ contains the videos shown during the final presentation
