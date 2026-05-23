.PHONY: install preprocess train test run all clean

install:
	pip install -r requirements.txt

preprocess:
	python data/preprocess.py

train:
	python models/train.py

test:
	pytest tests/ -v --tb=short

run:
	streamlit run app/streamlit_app.py

all: install preprocess train run

clean:
	rm -f data/*.parquet data/scaler.pkl
	rm -f models/*.pkl models/*.csv models/*.png
	find . -type d -name __pycache__ -exec rm -rf {} +
