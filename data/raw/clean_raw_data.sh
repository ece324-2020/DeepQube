folder=$1

for d in $folder/*/ ; do
    cd "$d"
    rm -rf *.txt *.zip Printing/ Interchange/*.html Interchange/*.jsonp Interchange/txt/
    cd ../..
done
