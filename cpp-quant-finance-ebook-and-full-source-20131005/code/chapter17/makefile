# Written for Mac OSX - 10.7.4

chap17: main.cpp fdm.o pde.o option.o payoff.o
	clang++ -o chap17 main.cpp fdm.o pde.o option.o payoff.o -arch x86_64 

fdm.o: fdm.cpp pde.o option.o payoff.o
	clang++ -c fdm.cpp pde.o option.o payoff.o -arch x86_64

pde.o: pde.cpp option.o payoff.o
	clang++ -c pde.cpp option.o payoff.o -arch x86_64

option.o: option.cpp payoff.o
	clang++ -c option.cpp payoff.o -arch x86_64

payoff.o: payoff.cpp
	clang++ -c payoff.cpp -arch x86_64