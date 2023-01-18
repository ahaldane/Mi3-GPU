
function! AlignMSA#AlignMSA()
    let l = getline('.')
	"let name = substitute(l, '^[^ ]* *', '', '')
    "
    let seqstart = max([0, matchend(l, '^[^ ]* *')])
	let s = l[seqstart:col('.')-1]
	let col = len(substitute(s, '[^A-Z-]', '', 'g'))

    let pat = '^\([^ ]*\) *\(\([a-z]*[A-Z-]\)\{'.col.'}\)'

    " first pass to determine prefixlen
    let matches = map(getline(1,'$'), {k,v -> {p->[len(p[1]), len(p[2])]}(matchlist(v, pat))})
    let namelen = max(map(copy(matches), {k,v -> v[0]}))
    let prelen = max(map(copy(matches), {k,v -> v[1]}))
    let extent = namelen + 1 + prelen

    " second pass to perform substitution (reuse above regex call somehow?)
    let sub = '\=submatch(1) . repeat(" ", '.extent.'-strlen(submatch(1)) - strlen(submatch(2))) . submatch(2)'
    call setline(1, map(getline(1,'$'), {k,v -> substitute(v, pat, sub, '')}))

    exe ':normal! '.extent.'|'
endfunction
