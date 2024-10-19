build:
	g++ $(F) `pkg-config opencv --cflags --libs`

clean:
	rm -f a.out

