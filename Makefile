.PHONY: all submit
competition = lish-moa

all: model/*.py | data/*.csv
	solution

submit:
	solution
	kaggle competitions submit -c $(competition) -f submission.csv -m "$(message)"

	sleep 2
	kaggle competitions submissions -c $(competition)

%.csv:
	kaggle competitions download -c $(competition) -p data/
