lr=(1e-1 1e-2 1e-3 1e-4 1e-5)

for lr in ${lr[@]}; do
	python3 trainModel.py -modelName m_lr=$lr -data datasets/Train/data.bin -target datasets/Train/labels.bin --lr $lr &
done;