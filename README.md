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
│   │   ├── agent.py
│   │   └── lspi.py
│   ├── envs                --- blackjack environments   
│   │   └── eleven_ace.py
│   ├── script              --- executable scripts
│   │   ├── benchmark.py
│   │   └── lspi_learn.py
│   └── utils
│       └── typedef.py
├── setup.py
└── test                    --- tests (pytest)
    ├── test_environment.py
    ├── test_hand.py
    └── test_lspi.py

```
