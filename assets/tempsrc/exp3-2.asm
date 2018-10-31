DATAS SEGMENT
    buf db 'tt$' 
DATAS ENDS

STACKS SEGMENT
    ;此处输入堆栈段代码
STACKS ENDS

CODES SEGMENT
    ASSUME CS:CODES,DS:DATAS,SS:STACKS
START:
    MOV AX,DATAS
    MOV DS,AX
    mov CX,0
    mov al,'$'
    lea DI,buf
    
repeater:
	INC DI
	mov bl,[DI]
	mov al,'$'
    sub al,bl
    JNZ repeater
    
    mov dx,DI
    add dl,30H
    mov ah,02h
    int 21H
    
    MOV AH,4CH
    INT 21H
CODES ENDS
    END START
