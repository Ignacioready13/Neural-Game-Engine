# About the engine design

I designed the classical engine in a way that lends itself to training a neural game engine on limited resources. Concretely, my engine's functionality needs to be well-defined and well-constrained so that the neural network has an easier time predicting its behavior. The limited input which I allow from the user needs to lead to *learnable* changes in the environment.


## List of **constraints** and **freedoms** I designed for my engine:

(Here, where I refer to the *engine*, I am referring to the classical game engine I used to generate the training data).
Because my goal was to train a complex neural game engine on a single laptop GPU, I had to constrain the classical engine as much as possible (while retaining an aspect of user choice) so that I could train a reasonably coherent model on limited hardware. 

1. **Environment**: all scenes in this game take place in a Backrooms-style maze of hallways and rooms. That's the only environment the network needs to learn.
2. **Movement**: the view is always moving forwards, along the current viewing direction, at a constant pace. The network doesn't need to learn what it means for a camera to stop, move slowly, move quickly, or turn in place.
3. **Camera**: the camera may only look left and right; the network does not need to learn up-and-down camera movements. The left-and-right movement is not free, but is constrained by limited inputs described in (4).
4. **Input**: The only input the engine reacts to is the x-position of a user's clicks; this is the only true freedom of my engine. The player may steer towards any *reachable* doorway by clicking on it; the engine only accepts their navigational choice if the click clearly indicates a certain doorway and that doorway is reachable by a path that won't require camera movements over a certain 'sudden-ness' threshold. If the engine accepts the click, it steers the camera along a calculatedly smooth path through it. If the player does not click (or their click is rejected) the engine keeps heading towards the doorway most directly ahead of the camera.

By balancing constraints and freedoms, I have allowed the user to retain control over where they walk in my engine (the user still feels like they are exploring the environment themselves), but massively reduced the difficulty of training a neural network to emulate this engine.

## Rules of procedural generation

For me, the intrigue of a neural game engine lies in its inherent *unpredictability*. While fascinating, projects which aim to perfectly replicate a totally concrete game environment will ultimately just create a blurrier version of something that already exists. Instead, I aim to *encourage* this unpredictability and inventiveness to train a model which generates/hallucinates interesting, novel environments that couldn't have been made in a classical game engine. My first step towards this goal was procedural generation.

Because the view in my game is *always moving forwards*, there *always* needs to be an available doorway directly ahead of the player. With careful procedural generation rules, my engine can generate the world on the fly, as the player moves around.

There are two types of **spaces**: **rooms** and **hallways**.
Any **space** has 4 walls, a ceiling, and a floor. 90 degree angles; spaces are always rectangular prisms.
The only **spaces** that **exist at a given moment** are the *space that the player is standing in*, and *all spaces directly adjacent to the space the player is standing in*.
For instance, if a player is standing in a room A, and room A has three doorways, then there are three spaces (rooms or hallways) adjacent to room A, each of which is fully visible and accessible through the doorway joining it and room A. 

Let's say that room B, hall C, and room D are the three adjacent spaces. Room B, hall A, and room C must also have doorways, but their doorways (besides the one adjoining each to room A) are fully filled in with a black texture. 

As soon as the player steps into room B, the black doorways are lifted from all of room B's doorways, revealing *its* adjacent spaces, which the engine randomly selects right at that moment (with randomized dimensions and doorway locations). Hall C and room D are removed from memory. In this way, we can simultaneously eliminate the computational complexity of trying to generate many layers of rooms which must not overlap, and eliminate the graphical complexity of rendering many distant wall faces we may never reach. That way our engine can generate the training data quickly and efficiently.

So, as the player traverses their environment, rooms and hallways are removed and added on-demand with no regard for where something *used* to be; there is no such thing as backtracking. By basing the engine off of random number generation, I aimed to introduce the same intrigue and 'exploration-based' environment into the neural model itself.

## Current status

Model flow96e, currently the best model, is able to consistently give the impression of being in a series of hallways and rooms (for the most part, it avoids devolving into static). It's able to show fluorescent lights passing by on the ceiling and occasionally able to hallucinate doorways in blank walls, but it lacks the ability to invent a distinct feature like a doorway or hallway bend and then maintain it until the camera passes it. For the most part, it always shows motion down a long forward-facing hallway or towards a morphing blank wall, and clicking has little noticable effect.

## Future goals

In the short-term, future models should do a better job generating concrete, recognizable features that move past the player as explained above. Future models should also be more responsive to clicking and avoid 'stopping in front of a blank wall' like flow96e does.

In the long-term, once future models can consistently turn down hallway bends and show well-defined features like doorways, I'd like to focus on adding to the 'inventiveness' aspects of the models. I've had the idea to randomly pull textures and models from the internet and display them on random walls and in random corners during training, and see what interesting combinations of shapes and colors the model learns to show during runtime.

