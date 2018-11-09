DATAS SEGMENT
    number dw 62897
    prime db 'Prime$'
    notprime db 'Not prime$'
DATAS ENDS

STACKS SEGMENT
    ;此处输入堆栈段代码
STACKS ENDS

CODES SEGMENT
    ASSUME CS:CODES,DS:DATAS,SS:STACKS
START:
    MOV AX,DATAS
    MOV DS,AX
    lea SI,number
    mov AX,[SI]
    mov BX,0002h
    mov DX,0
    div BX
    mov CX,AX
    mov AX,[SI]
l:
	mov DX,0
	div	BX
	cmp DX,0
	JZ zero
	inc BX
	mov AX,[SI]
	loop l
	lea DX,prime
	mov AH,09H
	int 21H
exit:   
    MOV AH,4CH
    INT 21H
    
zero:
	lea DX,notprime
	mov AH,09H
	int 21H
	JMP exit

CODES ENDS
    END START

