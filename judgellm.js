import Anthropic from "@anthropic-ai/sdk";
import 'dotenv/config'; 
import { OpenAI } from 'openai';
import { GoogleGenAI } from "@google/genai";
import readline from 'node:readline/promises';
import {stdin as input, stdout as output} from 'node:process'

let MAX_TOKENS = 200
let SYSTEM_PROMPT = `You are a Senior Software Developer. 
                    You can only answer questions from the field of Software and Computer Science.
                    Answer in maximum ${MAX_TOKENS} tokens`;

let  combinedSystemPrompt = `You are a Senior Software Developer. 
Your role is to analyze responses from different LLMs to a user asked question.
Each LLM Response begines with the Name of LLM Name Response Start and Ends with a Response Ends statements.
You will be provided with the Original user Question asked, and the LLM responses.
You will to analyze all LLM Responses and come up with the best response after the analysis.
Your Answer will be limited to the responses provided by the LLMs. 
Do not add any new information beyond what is provided in LLM response.
Your Response should start with Judge LLM Response Start: and Ends With Judge LLM Response Ends`


const claudeClient = new Anthropic();
const openaiClient = new OpenAI();
const gemini_ai = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY });
                    
const history = [];


async function askAllModels(userInput){

    const message = [...history, {role:'user',content:userInput}];
    const gemiMessages = message.map(m => ({
        role: m.role === 'assistant' ? 'model' : 'user',
        parts: [{ text: m.content }],
    }));   
    
    const [claude_response, openai_response, gemini_response]= await Promise.all([

        claudeClient.messages.create({
            model: 'claude-opus-4-8',
            max_tokens: MAX_TOKENS,
            messages: message,
            system: SYSTEM_PROMPT,
        }),
        openaiClient.responses.create({
            model: 'gpt-4o-mini',
            input: message,
            instructions: SYSTEM_PROMPT,
        }),
        gemini_ai.models.generateContent({
            model: 'gemini-flash-latest',
            contents: gemiMessages,
            config: { systemInstruction: SYSTEM_PROMPT },
        }),
    ])
    let claudeResponse = '';
    for (const block of claude_response.content) {
        if (block.type === 'text') claudeResponse = block.text;
    }
    const openaiResponse = openai_response.output_text;
    const googleResponse = gemini_response.text;
    const combinedResponse = `
    User Question: ${userInput}
    
    LLM1 Response Start:
    ${claudeResponse}
    LLM1 Response End.
    
    LLM2 Response Start:
    ${openaiResponse}
    LLM2 Response End.
    
    LLM3 Response Start:
    ${googleResponse}
    LLM3 Response End.`;


    
    const judgeResult = await openaiClient.responses
    .create({
        model : 'o4-mini-2025-04-16',
        input:  [...history, { role: 'user', content: combinedResponse }],
        instructions: combinedSystemPrompt
    });
    console.log('\n'+ combinedResponse);

    const judgeAnswer = judgeResult.output_text;

    // Save clean turns for the next question
    history.push({ role: 'user', content: userInput });
    history.push({ role: 'assistant', content: judgeAnswer });

    return judgeAnswer;

}//askAllModels


const r1 = readline.createInterface({input,output});

while(true){
    const userInput = (await r1.question('Ask a question or \'exit\'')).trim();
    if(!userInput) continue;
    if(userInput.toLowerCase() === 'exit') break;
    try{
        const answer = await askAllModels(userInput)
        console.log('\n'+answer);
    }
    catch(e)
    {
        console.log('Error: ',e.message)
    }
}

r1.close();