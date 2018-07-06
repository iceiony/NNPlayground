call plug#begin('~/.vim/plugged')
" Plug 'kchmck/vim-coffee-script'
Plug 'klen/python-mode'
Plug 'jalvesaq/nvim-r' 
Plug 'ervandew/screen'
" Plug 'Valloric/YouCompleteMe'
Plug 'Shougo/deoplete.nvim', { 'do': ':UpdateRemotePlugins' }
Plug 'altercation/vim-colors-solarized'
Plug 'scrooloose/nerdtree'
Plug 'kassio/neoterm'
" Plug 'isRuslan/vim-es6'
" Plug 'hkupty/iron.nvim'
" Plug 'moll/vim-node'
" Plug 'mustache/vim-mustache-handlebars'
call plug#end()

let maplocalleader=','
let mapleader=','

filetype plugin on
let g:deoplete#enable_at_startup = 1

set nu
syntax on
set autoindent
set smartindent
set incsearch
set expandtab
set tabstop=4
set shiftwidth=4

set incsearch     
set hlsearch      
set mouse=n       
set ruler         
set ttyfast       
set cursorline    
set showmatch     
set enc=utf-8     
set scrolloff=5  

set background=dark
let g:solarized_termcolors = 256
colorscheme solarized

let R_show_args = 1                                                                              
let R_args_in_stline = 1     

autocmd FileType r set tags+=./tags

set completeopt-=preview

" Python Mode Settings
let g:pymode_python = 'python3'
let g:pymode_rope = 0
 
" " Documentation
let g:pymode_doc = 0
" let g:pymode_doc_key = 'K'
" 
" "Linting
let g:pymode_lint = 0
" let g:pymode_lint_checker = "pyflakes,pep8"
" " Auto check on save
" let g:pymode_lint_write = 0
" 
" " Support virtualenv
" let g:pymode_virtualenv = 0
" 
" " Enable breakpoints plugin
let g:pymode_breakpoint = 0
" let g:pymode_breakpoint_bind = '<leader>b'
" 
" " syntax highlighting
 let g:pymode_syntax = 0
 let g:pymode_syntax_all = 0
" let g:pymode_syntax_indent_errors = g:pymode_syntax_all
" let g:pymode_syntax_space_errors = g:pymode_syntax_all
map <C-n> :NERDTreeToggle<CR>
tnoremap <Esc> <C-\><C-n>       

" Don't autofold code
let g:pymode_folding = 0

" NeoTerm
nnoremap <silent> <localleader>aa :TREPLSendFile<cr>
nnoremap <silent> <localleader>l :TREPLSendLine<cr>
vnoremap <silent> <localleader>ss :TREPLSendSelection<cr>
