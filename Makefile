dirs = R/ B/ V/ I/
targets :=$(addsuffix final, $(dirs)) $(addsuffix result_sum.fts, $(dirs)) 

all: $(dirs) $(targets)

%/:
	mkdir -p $@

%/result_sum.fts: start.py
	python3 start.py $(patsubst %/result_sum.fts,%,$@)

%/final: %/result_sum.fts 
	python3 photometry.py $(patsubst %/final,%,$@)

.PHONY: all 


