## Describe the effect each of the P, I, D components had in your implementation
P: through proportional parameter to pull vehicle back to center of the road
I: reduce oscillation of driving like a drunk driver
D: avoid overshooting and undershooting of setpoints
## Describe how the final hyperparameters were chosen.
1) first set I and D to zero, and find P boundary value that vehicle start to oscillate
2) gradually increase I value to make the oscillation in a reasonable region
3) gradually increase D value to make steering stable