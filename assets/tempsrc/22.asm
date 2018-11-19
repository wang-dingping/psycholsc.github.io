DATAS SEGMENT
    string db '0123456789ABCDEF'
    EOL db 0AH,0DH,'$' ;10,13
DATAS ENDS

STACKS SEGMENT
STACKS ENDS

CODES SEGMENT
    ASSUME CS:CODES,DS:DATAS,SS:STACKS
START:
    MOV AX,DATAS
    MOV DS,AX
beg:
    mov AH,01H		; input
    int 21H
    lea DX,EOL
    mov AH,09H		; output EOL
    int 21H
    cmp AL,'a'		; a?
    JZ prebuf1		; yep
    cmp AL,'b'		; b?
    JZ prebuf2
    cmp AL,'c'
    JZ prebuf3
    cmp AL,'d'
    JZ prebuf4
    cmp AL,1BH		; esc?	esc == 1BH
    JZ preexit
    JMP beg			; unlimited loop

prebuf1:
	call buf1		; just a little segment, not a proc. jmp to ret
	jmp beg
prebuf2:
	call buf2
	jmp beg
prebuf3:
	call buf3
	jmp beg
prebuf4:
	call buf4
	jmp beg    
preexit:
	call exit
	jmp beg
buf1 proc near		; a proc. can ret. can also jmp to ret.
	lea DX,buf1
	call print
	jmp beg
	buf1 endp
	
buf2 proc near
	lea DX,buf2
	call print
	jmp beg
	buf2 endp
buf3 proc near
	lea DX,buf3
	call print
	jmp beg
	buf3 endp
buf4 proc near
	lea DX,buf4
	call print
	jmp beg
	buf4 endp
exit proc near
	MOV AH,4CH
    INT 21H
    exit endp 
print proc near
	; DX = index 
	mov CX,16
ll: 				;xlat
	JCXZ con2
	sub CX,4
	mov BX,DX
	push DX
	shr BX,CL
	and BX,000FH
	mov SI,BX
	mov DL,[SI]
	mov AH,02H
	int 21H
	pop DX
	JMP ll
con2:			
	lea DX,EOL
	mov AH,09H
	int 21H
	ret
	print endp
CODES ENDS
    END START
; longest ever.


