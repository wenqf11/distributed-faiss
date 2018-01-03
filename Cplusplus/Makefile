
CC=g++

CFLAGS=-c -Wall -lboost_system -pthread -std=c++11

all: server

server: 
	$(CC) $(CFLAGS) server.cpp -o server 

clean: 
	rm -rf *o server
