gcl@gcl-K45VD:~/number_5_25$ ls -l *.JPG|xargs -n 1 bash -c 'convert "$0" "${0%.JPG}.jpg"'
