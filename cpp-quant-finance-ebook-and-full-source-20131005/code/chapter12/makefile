# Written for Mac OSX - 10.7.4

chap12: main.cpp asian.o payoff.o
	clang++ -o chap12 main.cpp asian.o payoff.o -arch x86_64

asian.o: asian.cpp payoff.o
	clang++ -c asian.cpp payoff.o -arch x86_64 

payoff.o: payoff.cpp
	clang++ -c payoff.cpp -arch x86_64
