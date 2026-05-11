.PHONY: build clean clean-build


build:
	cmake -S . -B build
	cmake --build build

clean:
	-rm -rf build

clean-build: clean
	cmake -S . -B build
	cmake --build build
