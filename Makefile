.PHONY: all submit

competition = lish-moa


local:
	python models/kmlp.py && echo "\a"

submit:
	kaggle competitions submit -c $(competition) -f submission.csv -m "$(message)"

	sleep 2
	kaggle competitions submissions -c $(competition)

submission.csv: model/*.py  data/train_targets_nonscored.csv
	solution

%.csv: data/$(competition).zip
	unzip $< -d data/

data/$(competition).zip:
	kaggle competitions download -c $(competition) -p data/
