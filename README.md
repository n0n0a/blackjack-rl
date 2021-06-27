# ♦blackjack-rl♦
blackjack-rl offers **blackjack environment** and **reinforcment learning method** for it.
We first use value-based Q-Learning as solver method.

# ♤How to install
Clone this repository, and excecute below command
```shell
$ make install
```
*Required Python3.9~

# ♤How to learn
```shell
$ cd blackjack_rl/script
$ python lspi_learn.py
```

# Direcotory structure
```shell
blackjack-rl
├── LICENSE
├── Makefile
├── README.md
├── blackjack_rl
│   ├── agent               --- blackjack agents
│   ├── envs                --- blackjack environments   
│   ├── script              --- executable scripts
│   └── utils               --- utilities
├── setup.py
└── test                    --- tests (pytest)

```
