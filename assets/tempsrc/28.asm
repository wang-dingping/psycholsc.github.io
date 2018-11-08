DATAS SEGMENT
    array dw 1234H,5678H,2345H,3456H
    len equ ($-array)/2
    MAX dw 0
    MIN dw 0
DATAS ENDS

STACKS SEGMENT
    ;此处输入堆栈段代码
STACKS ENDS

CODES SEGMENT
    ASSUME CS:CODES,DS:DATAS,SS:STACKS
START:
    MOV AX,DATAS
    MOV DS,AX
    mov CX,len
    mov SI,0
    mov AX,[SI]
    mov MAX,AX
l:
	inc SI
	inc SI
	mov AX,[SI]
	cmp MAX,AX
	JL se1
con:	
	loop l
	mov SI,0
	mov BX,[SI]
	mov MIN,BX
l2:
	inc SI
	inc SI
	mov BX,[SI]
	cmp MIN,BX
	JG se2
con2:
	loop l2
    
    MOV AH,4CH
    INT 21H
se1:
	mov AX,[SI]
	mov MAX,AX
	jmp con
se2:
	mov BX,[SI]
	mov MIN,BX
	jmp con2
CODES ENDS
    END START

