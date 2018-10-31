DATAS SEGMENT
    buf1 db 'how many &s are there in this &ny f&&king & string?'
    count equ $-buf1
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
    lea DI,buf1
    mov AL,'&'
    mov CX,count
    CLD
    mov DX,0
    
lp:
    repnz scasb
    INC CX
    INC DX
    DEC CX
    JNZ lp
    
    DEC DX
    ; answer is in DL.
    MOV AH,4CH
    INT 21H
CODES ENDS
    END START

