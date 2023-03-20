dirs = R/ B/ V/ I/
targets := result/R result/B result/V result/I

all: $(dirs) $(targets) 

%/:
	mkdir -p $@

result/%:
	python3 start.py $(notdir $@)

.PHONY: all 
