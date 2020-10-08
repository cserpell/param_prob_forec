#!/bin/bash
# Script to convert original UCI dataset into a nice CSV

# Exit immediately
set -e

AWK_SCRIPT=$(cat << EOM2
{
  if (started == 0) {
    printf("GMT,", \$1);
    for (i = 3; i < NF; i++) {
      printf("%s,", \$i);
    }
    printf("%s\n", \$NF);
    started = 1;
  } else {
    split(\$1, dat, "/");
    printf("%s-%s-%s %s,", dat[3], dat[2], dat[1], \$2);
    for (i = 3; i < NF; i++) {
      to_print = \$i;
      if (\$i == "?") {
        to_print = "";
      }
      printf("%s,", to_print);
    }
    printf("%s\n", \$NF);
  }
}
EOM2
)

awk -F';' "${AWK_SCRIPT}"
