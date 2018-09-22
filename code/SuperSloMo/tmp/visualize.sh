
for file in adobe_*.flo; do
	fname=${file%.*};
	./color_flow $file $fname.png
done
