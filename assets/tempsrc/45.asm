DATAS SEGMENT
    plain db 'input plain text$'
    len equ $-plain
    key db len/7+1 dup('ABXmv#7')
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
	xor AL,AH
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
CODES ENDS
    END START
