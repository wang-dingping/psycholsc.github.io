DATAS SEGMENT
    buf1 db 'this is a string'
    count equ $-buf1
    buf2 db count dup(0)
DATAS ENDS

STACKS SEGMENT
    
STACKS ENDS

CODES SEGMENT
    ASSUME CS:CODES,DS:DATAS,SS:STACKS
START:
    MOV AX,DATAS
    MOV DS,AX
    mov ES,AX
    lea SI,buf1
    lea DI,buf2
    mov cx,count
    rep movsb
    
    
    MOV AH,4CH
    INT 21H
CODES ENDS
    END START
