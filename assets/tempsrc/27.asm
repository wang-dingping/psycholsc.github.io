DATAS SEGMENT
    string db 'data,name,time,file,code,path,user,exit,quit,text,$'
    len equ $-string
    maxlen db 5,0,0,0,0,0,0
    EOL db 0AH,0DH,'$'
    
    
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
    mov AH,0AH
    lea DX,maxlen
    int 21H
    
    mov CX,5
    lea SI,string
    lea DX,maxlen
    add DX,2
    mov DI,DX
    mov DX,0
l:
	mov CX,5
	lea DI,maxlen
    add DI,2
    mov SI,0
    add SI,DX
	repz cmpsb
	cmp CX,0
	JNZ notzero
	JZ zero
con:
	JMP l
zero:	
	lea DX,EOL
	mov AH,09H
	int 21H
	
	mov DX,SI
	sub DX,5
	mov DI,DX
	mov CX,len
	sub CX,SI
	add CX,4
	;xchg SI,DI
	rep movsb
	lea DX,string
	mov AH,09H
	int 21H
exit:	    
    MOV AH,4CH
    INT 21H
notzero:
	inc DX
	cmp DX,len-4
	JZ exit
	JNZ con
	
CODES ENDS
    END START

