# Campbell-Cochrane Habit Model

This repository contains an implementation of the Campbell-Cochrane habit model for asset pricing, which assumes price-consumption ratio $\frac{P}{C}$ is a function $v$ of consumption surplus $S$. A dense neural network is trained to obtain the coordinates (i.e., optimal weights) of the projection of $v$ onto Chebyshev's polynomial space.

- `habit_model.py` is an implementation of the habit model class. 

- `early_stop.py` implements early stopping to prevent overfitting, which is the same as that used in https://github.com/Kaiwen-Hou-KHou/spectralRegularization. 

- `demo.ipynb` shows the reproducible training process and results (loss curves, weights matrix, and $\frac{P}{C}$ against $S$), and a bit of interpretations as well.




# References

Campbell, J. Y., & Cochrane, J. H. (1999). By force of habit: A consumption-based explanation of aggregate stock market behavior. <i>Journal of political Economy</i>, 107(2), 205-251.
