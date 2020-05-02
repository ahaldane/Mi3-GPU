" Vim syntax file
" Language: Protein amino-acid alignments
" Author: Allan Haldane, allan.haldane@gmail.com
"
" To use this vim syntax highlighting:
" Put this file in you .vim/syntax directory. Then when you open an Mi3-GPU
" formatted protein MSA in vim, type ":setf msa" to highlight the MSA.

if exists("b:current_syntax")
  finish
endif

set nowrap
set go=gmrLtTb

" Sequence:
for c in split('ACDEFGHIKLMNPQRSTVWY', '\zs')
  execute "syn match prt" . c . " \"[" . c . "]\\+\""
  execute "syn match prtl" . tolower(c) . " \"[" . tolower(c) . "]\\+\""
endfor

" Comments:
syn match msaHead "^[^| \t]*[| \t]"
syn match msaComment "^[>#].*"

" synchronizing
syn sync maxlines=50

" Define the default highlighting
" see: view-source:http://www.jalview.org/version118/documentation.html#colour
highlight prtV ctermfg=black ctermbg=155
highlight prtI ctermfg=black ctermbg=155
highlight prtL ctermfg=black ctermbg=119
highlight prtF ctermfg=black ctermbg=85
highlight prtW ctermfg=black ctermbg=81
highlight prtH ctermfg=black ctermbg=111
highlight prtR ctermfg=black ctermbg=63
highlight prtK ctermfg=black ctermbg=135
highlight prtN ctermfg=black ctermbg=171
highlight prtQ ctermfg=black ctermbg=206
highlight prtE ctermfg=black ctermbg=205
highlight prtD ctermfg=black ctermbg=203
highlight prtS ctermfg=black ctermbg=209
highlight prtT ctermfg=black ctermbg=215
highlight prtG ctermfg=black ctermbg=215
highlight prtP ctermfg=black ctermbg=221
highlight prtC ctermfg=black ctermbg=227
highlight prtY ctermfg=black ctermbg=227
highlight prtA ctermfg=black ctermbg=white
highlight prtM ctermfg=black ctermbg=white
highlight prtlv cterm=italic ctermfg=gray ctermbg=155
highlight prtli cterm=italic ctermfg=gray ctermbg=155
highlight prtll cterm=italic ctermfg=gray ctermbg=119
highlight prtlf cterm=italic ctermfg=gray ctermbg=85
highlight prtlw cterm=italic ctermfg=gray ctermbg=81
highlight prtlh cterm=italic ctermfg=gray ctermbg=111
highlight prtlr cterm=italic ctermfg=gray ctermbg=63
highlight prtlk cterm=italic ctermfg=gray ctermbg=135
highlight prtln cterm=italic ctermfg=gray ctermbg=171
highlight prtlq cterm=italic ctermfg=gray ctermbg=206
highlight prtle cterm=italic ctermfg=gray ctermbg=205
highlight prtld cterm=italic ctermfg=gray ctermbg=203
highlight prtls cterm=italic ctermfg=gray ctermbg=209
highlight prtlt cterm=italic ctermfg=gray ctermbg=215
highlight prtlg cterm=italic ctermfg=gray ctermbg=215
highlight prtlp cterm=italic ctermfg=gray ctermbg=221
highlight prtlc cterm=italic ctermfg=gray ctermbg=227
highlight prtly cterm=italic ctermfg=gray ctermbg=227
highlight prtla cterm=italic ctermfg=gray ctermbg=white
highlight prtlm cterm=italic ctermfg=gray ctermbg=white
highlight msaComment ctermfg=None ctermbg=None
highlight msaHead ctermfg=None ctermbg=None

let b:current_syntax = "msa"
" vim: ts=8
