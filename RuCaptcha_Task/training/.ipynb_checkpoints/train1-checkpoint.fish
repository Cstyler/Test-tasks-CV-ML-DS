cd /srv/data_science/training/code/price_ocr/grcnn
set script_name $argv[1]
/home/agazade/.conda/envs/main/bin/python $script_name > output.txt 2> errors.txt
