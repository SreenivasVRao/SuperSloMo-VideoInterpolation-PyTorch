
for file in *.flo; do
	fname=${file%.*};
	./color_flow $file $fname.png
done
