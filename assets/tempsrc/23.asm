DATAS SEGMENT
    hex db '0123456789ABCDEF'
    EOL db 0AH,0DH,'$'
    err db 'No such command!',0AH,0DH,'$'
DATAS ENDS

STACKS SEGMENT
    ;此处输入堆栈段代码
STACKS ENDS

CODES SEGMENT
    ASSUME CS:CODES,DS:DATAS,SS:STACKS
START:
    MOV AX,DATAS
    MOV DS,AX
    
l:
	mov DX,0
    mov AH,01H
    int 21H
    lea DX,EOL
    mov AH,09H
    int 21H
    cmp AL,'0'
    JZ seg0
    cmp AL,'1'
    JZ sega
    cmp AL,'2'
    JZ segb
    cmp AL,'3'
    JZ segc
    cmp AL,'4'
    JZ segd
    cmp AL,0DH
    JZ exit
    lea DX,err
    mov AH,09H
    int 21H
    jmp l
con:
	lea DX,EOL
	mov AH,09H
	int 21H
    JMP l
    
    
    
exit:    
    MOV AH,4CH
    INT 21H
seg0:
	call buf0
	JMP con
sega:
	call buf1
	JMP con
segb:
	call buf2
	jmp con
segc:
	call buf3
	jmp con
segd:
	call buf4
	jmp con
buf0 proc near
	lea DX,buf0
	call print
	ret
buf0 endp
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
	mov CX,16
lo:
	push DX
	sub CX,4
	shr DX,CL
	and DX,000FH
	mov DI,DX
	mov DL,[DI]
	mov AH,02H
	int 21H
	pop DX
	cmp CX,0
	JZ re
	jmp lo
re:
	ret
print endp

CODES ENDS
    END START
