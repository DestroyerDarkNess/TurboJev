declare module '@huggingface/transformers' {
  export const Tensor: any;
  export const AutoTokenizer: { from_pretrained(model:string, options?:any): Promise<any> };
  export const AutoModelForCausalLM: { from_pretrained(model:string, options?:any): Promise<any> };
}
