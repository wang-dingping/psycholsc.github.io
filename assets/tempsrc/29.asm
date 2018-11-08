DATAS SEGMENT
    buffer dw 1234H,2345H,3456H,4567H,5678H,6789H,789AH,89ABH,9ABCH,0ABCDH
	len equ ($-buffer)/2
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
    mov SI,0
    mov CX,len
l0:
	mov AX,[SI]
	inc SI
	inc SI
	cmp AX,0
	JL lower
	JNL l
con:
	lea DX,EOL
	mov AH,09H
	int 21H
	loop l0
	JMP exit

lower:
	neg AX
l:
	mov BX,10
	mov DX,-1
	push DX
re:
	xor DX,DX
	div BX
	push DX
	test AX,AX
	jne re
	mov BX,-1
	mov AH,2
disp:
	pop DX
	cmp DX,BX
	je con
	add dl,'0'
	int 21H
	jmp disp
	jmp con
	
exit:
    MOV AH,4CH
    INT 21H
CODES ENDS
    END START

