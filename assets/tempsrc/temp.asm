DATAS SEGMENT
    string db '0123456789ABCDEF'
    EOL db 10,13,'$'
DATAS ENDS

STACKS SEGMENT
    ;此处输入堆栈段代码
STACKS ENDS

CODES SEGMENT
    ASSUME CS:CODES,DS:DATAS,SS:STACKS
START:
    MOV AX,DATAS
    MOV DS,AX
    mov ES,AX
    mov SI,0
    mov DI,0
L:    
    mov AH,01H
    int 21H
    lea DX,EOL
	mov AH,09H
	int 21H
    cmp AL,'a'
    JZ prebuf1
    cmp AL,'b'
    JZ prebuf2
    cmp AL,'c'
    JZ prebuf3
    cmp AL,'d'
    JZ prebuf4
    cmp AL,1BH
    JZ exit
    JMP L
    
    
exit:
    MOV AH,4CH
    INT 21H

prebuf1 proc near
	call buf1
	jmp L
prebuf1 endp
prebuf2 proc near
	call buf2
	jmp L
prebuf2 endp
prebuf3 proc near
	call buf3
	jmp L
prebuf3 endp
prebuf4 proc near
	call buf4
	jmp L
prebuf4 endp
buf1 proc near
	lea DX,buf1
	call print
	ret
buf1 endp

buf2 proc near
	lea DX,buf2
	call print
	ret
buf2 endp

buf3 proc near
	lea DX,buf3
	call print
	ret
buf3 endp

buf4 proc near
	lea DX,buf4
	call print
	ret
buf4 endp



print proc near
	mov CL,16
stt:
	push DX
	sub CL,4
	shr DX,CL
	and DX,000FH
	mov SI,DX
	mov DL,[SI]
	mov AH,02H
	int 21H
	pop DX
	cmp CL,0
	JNZ stt
	lea DX,EOL
	mov AH,09H
	int 21H
	ret
print endp
CODES ENDS
    END START
