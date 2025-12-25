# Force XeLaTeX even when latexmk is invoked with -pdf
$pdf_mode = 5;                 # pdf output via xelatex
$pdflatex = 'xelatex %O %S';   # latexmk uses this variable for -pdf mode
$biber    = 'biber %O %B';
