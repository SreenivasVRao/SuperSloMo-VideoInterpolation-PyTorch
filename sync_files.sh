set -e;
echo "Sync code..."
rsync -rL scripts/ sreenivasv@gypsum.cs.umass.edu:/home/sreenivasv/CS701/VideoInterpolation-PyTorch/scripts/
# rsync -rL data/ sreenivasv@gypsum.cs.umass.edu:/home/sreenivasv/CS701/VideoInterpolation-PyTorch/data/
rsync -rL weights/ sreenivasv@gypsum.cs.umass.edu:/home/sreenivasv/CS701/VideoInterpolation-PyTorch/weights/
rsync -rL README.MD sreenivasv@gypsum.cs.umass.edu:/home/sreenivasv/CS701/VideoInterpolation-PyTorch/README.MD
rsync -rL sync_files.sh sreenivasv@gypsum.cs.umass.edu:/home/sreenivasv/CS701/VideoInterpolation-PyTorch/sync_files.sh
echo "Sync configs..."
rsync -rL configs/ sreenivasv@gypsum.cs.umass.edu:/home/sreenivasv/CS701/VideoInterpolation-PyTorch/configs/
echo "Sync git..."
# rsync -rL .git/ sreenivasv@gypsum.cs.umass.edu:/home/sreenivasv/CS701/VideoInterpolation-PyTorch/.git/
echo "Done!"
exit 0;
