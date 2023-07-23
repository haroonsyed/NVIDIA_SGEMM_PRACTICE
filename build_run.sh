rm sgemm
rm -r build
mkdir build
cd ./build
cmake ..
make
cd ..
./run.sh