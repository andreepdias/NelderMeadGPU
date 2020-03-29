# filename=ARG1
# fileoutput=ARG2

# set terminal png notransparent rounded giant font "/usr/share/fonts/msttcore/arial.ttf" 24 \
  # size 1200,960 # set output fileoutput

set tics nomirror

set style line 11 lc rgb '#808080' lt 0
set border 3 back ls 11

set style line 12 lc rgb '#808080' lt 0 lw 1
set grid back ls 12

set style line 1 lc rgb '#8b1a0e' pt 1 ps 1.0 lt 1 lw 2 # --- red
set style line 2 lc rgb '#5e9c36' pt 6 ps 1.0 lt 1 lw 2 # --- green

set key top right

set xlabel 'evaluations number'
set ylabel 'objective function value'
set xrange [0:500500]
# set yrange [-20:-10]
set autoscale y

plot "output_13_2EWH.txt" u 1:2 t "2EWH" w lp ls 1
#, \
#    filename u 1:3 t "Average" w lp ls 2
pause 0.5
reread