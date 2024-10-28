build:
	g++ $(F) `pkg-config opencv4 --cflags --libs`

clean:
	rm -f a.out

test:
	make build F=incl_test.cc && ./a.out
