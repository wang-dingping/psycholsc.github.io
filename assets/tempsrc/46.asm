DATAS SEGMENT
    plain db 'input plain text$'
    len equ $-plain
    key db len/10+1 dup(-2,4,1,0,-3,5,2,-4,-4,6)
    endf db 0AH,0DH,'$'
    enc db len dup(0)
    
DATAS ENDS

STACKS SEGMENT
    ;此处输入堆栈段代码
STACKS ENDS

CODES SEGMENT
    ASSUME CS:CODES,DS:DATAS,SS:STACKS
START:
    MOV AX,DATAS
    MOV DS,AX
    
    ;lea DI,enc
    ;mov DL,20H
    ;mov [DI],DL
    mov CX,len
    lea SI,plain
    lea DI,key
    mov DX,0
lo:
	mov AL,[SI]
	mov AH,[DI]
	push DX
	mov DL,[DI]
	test DL,80H
	JNZ se1;-
	JZ se2;+
con:	
	;xor AL,AH
	pop DX
	lea BX,enc
	add BX,DX
	inc DX
	xchg BX,DI
	mov [DI],AL
	xchg BX,DI
	inc DI
	inc SI
	loop lo

    MOV AH,4CH
    INT 21H
se1:
	neg DL
	push CX
	mov CL,DL
	rol AL,CL
	pop CX
	jmp con
se2:
	push CX
	mov CL,DL
	ror AL,CL
	pop CX
	jmp con
	
CODES ENDS
    END START

