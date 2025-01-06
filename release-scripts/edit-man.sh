#! /bin/bash

cd "$(dirname $0)"
cd ../man/man1

for file in $(/bin/ls -1 *.1)
do 
  # On Linux
  #sed -i -e 's|3 July 2023|6 January 2025|' -e 's|"1.1"|"1.2"|' $file
  # On macOS
  sed -I '' -e 's|3 July 2023|6 January 2025|' -e 's|"1.1"|"1.2"|' $file
done
