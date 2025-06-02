import os
import sys
import json
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import click
from dotenv import load_dotenv
import openai
from pydantic import BaseModel, Field
import re

load_dotenv()

class LTLFormula(BaseModel):
    formula: str = Field(description="The LTL formula")
    inputs: List[str] = Field(description="List of input signals")
    outputs: List[str] = Field(description="List of output signals")
    description: str = Field(description="Human-readable description of what the formula specifies")

class NL2Verilog:
    
    def __init__(self, api_key: str, model: str = "gpt-4-turbo-preview", verbose: bool = False):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.verbose = verbose
        
    def log(self, message: str, level: str = "INFO"):
        if self.verbose or level == "ERROR":
            print(f"[{level}] {message}")
    
    def clean_ltl_formula(self, formula: str, inputs: List[str], outputs: List[str]) -> str:
        all_props = inputs + outputs
        cleaned = formula
        self.log(f"Original formula: {formula}")
        self.log(f"Cleaned formula: {cleaned}")
        return cleaned
    
    def nl_to_ltl(self, specification: str) -> LTLFormula:
        self.log("Converting natural language to LTL...")
        system_prompt = """You are an expert in formal verification and temporal logic. 
Convert natural language hardware specifications into Linear Temporal Logic (LTL) formulas that work with ltlsynt.

Rules:
1. Use standard LTL operators: G (globally/always), F (eventually), X (next), U (until), R (release)
2. Use & for AND, | for OR, ! for NOT, -> for implication, <-> for equivalence
3. Identify and list all input and output signals
4. Keep formulas as simple as possible for ltlsynt compatibility
5. Return a JSON object with: formula (as a single string), inputs, outputs, and description
6. The formula field MUST be a single string, NOT an array

Key Patterns for ltlsynt:
- Simple implication: G(input -> X(output))
- Eventually: G(input -> F(output))
- Toggle: G(output <-> X(!output))
- One-cycle delay: G((input -> X(output)) & (!input -> X(!output)))
- Conditional output: G(condition -> X(output))

Examples:
- "Output follows input" → G(input -> X(output))
- "Output toggles every cycle" → G(output <-> X(!output))
- "Request eventually gets grant" → G(request -> F(grant))
- "Output equals previous input" → G((input -> X(output)) & (!input -> X(!output)))
- "Button triggers output" → G(button -> X(output))
- "Reset clears output" → G(reset -> X(!output))

For complex behaviors, combine multiple conditions into a single formula using & operator."""

        user_prompt = f"""Convert this specification to LTL:
"{specification}"

Return a JSON object with:
- formula: the LTL formula
- inputs: list of input signal names
- outputs: list of output signal names  
- description: what the formula specifies"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1
            )
            
            result = json.loads(response.choices[0].message.content)
            ltl_formula = LTLFormula(**result)
            
            self.log(f"Generated LTL: {ltl_formula.formula}")
            self.log(f"Inputs: {', '.join(ltl_formula.inputs)}")
            self.log(f"Outputs: {', '.join(ltl_formula.outputs)}")
            
            return ltl_formula
            
        except Exception as e:
            self.log(f"Error in GPT-4 conversion: {str(e)}", "ERROR")
            raise
    
    def create_tlsf(self, ltl_formula: LTLFormula, output_file: str) -> None:
        self.log("Creating TLSF file...")
        
        tlsf_content = f"""INFO {{
  TITLE:       "Generated from: {ltl_formula.description}"
  DESCRIPTION: "{ltl_formula.description}"
  SEMANTICS:   Mealy
  TARGET:      Mealy
}}

MAIN {{
  INPUTS {{
    {';'.join(ltl_formula.inputs)};
  }}
  
  OUTPUTS {{
    {';'.join(ltl_formula.outputs)};
  }}
  
  ASSERT {{
    {ltl_formula.formula};
  }}
}}
"""
        
        with open(output_file, 'w') as f:
            f.write(tlsf_content)
        
        self.log(f"Created TLSF file: {output_file}")
    
    def run_command(self, command: List[str], cwd: Optional[str] = None) -> Tuple[int, str, str]:
        self.log(f"Running: {' '.join(command)}")
        
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                cwd=cwd
            )
            return result.returncode, result.stdout, result.stderr
        except FileNotFoundError:
            return -1, "", f"Command not found: {command[0]}"
    
    def ltl_to_aiger(self, ltl_formula: LTLFormula, output_dir: str) -> Optional[str]:
        self.log("Converting LTL to AIGER using ltlsynt...")
        
        base_name = Path(output_dir).joinpath("specification")
        aag_file = str(base_name.with_suffix('.aag'))
        
        ins = ','.join(ltl_formula.inputs) if ltl_formula.inputs else ""
        outs = ','.join(ltl_formula.outputs)
        
        cmd = ["ltlsynt", 
               "--formula", ltl_formula.formula,
               "--ins", ins,
               "--outs", outs,
               "--aiger"]
        
        exit_code, stdout, stderr = self.run_command(cmd)
        
        if exit_code != 0:
            self.log(f"ltlsynt error: {stderr}", "ERROR")
            return None
        
        if stdout.startswith("REALIZABLE"):
            self.log("Formula is REALIZABLE")
            aag_content = '\n'.join(stdout.split('\n')[1:])
            with open(aag_file, 'w') as f:
                f.write(aag_content)
        elif stdout.startswith("UNREALIZABLE"):
            self.log("Formula is UNREALIZABLE", "ERROR")
            return None
        else:
            with open(aag_file, 'w') as f:
                f.write(stdout)
        
        if not os.path.exists(aag_file) or os.path.getsize(aag_file) == 0:
            self.log("AIGER file not created or empty", "ERROR")
            return None
            
        self.log(f"Created AIGER file: {aag_file}")
        return aag_file
    
    def aiger_to_verilog(self, aag_file: str, output_dir: str) -> Optional[str]:
        self.log("Converting AIGER to Verilog...")
        
        base_name = Path(aag_file).stem
        aig_file = os.path.join(output_dir, f"{base_name}.aig")
        verilog_file = os.path.join(output_dir, f"{base_name}.v")
        
        exit_code, stdout, stderr = self.run_command(
            ["aigtoaig", aag_file, aig_file]
        )
        
        if exit_code != 0:
            self.log(f"aigtoaig error: {stderr}", "ERROR")
            return None
        
        abc_command = f'read_aiger {aig_file}; write_verilog {verilog_file}'
        exit_code, stdout, stderr = self.run_command(
            ["abc", "-c", abc_command]
        )
        
        if exit_code != 0:
            self.log(f"ABC error: {stderr}", "ERROR")
            return None
        
        if not os.path.exists(verilog_file):
            self.log("Verilog file not created", "ERROR")
            return None
            
        self.log(f"Created Verilog file: {verilog_file}")
        return verilog_file
    
    def post_process_verilog(self, verilog_file: str, ltl_formula: LTLFormula) -> str:
        self.log("Post-processing Verilog...")
        
        with open(verilog_file, 'r') as f:
            verilog_content = f.read()
        
        verilog_content = re.sub(r'module\s+\\[^\s]+\s+\(', 'module generated_module (', verilog_content)
        
        header = f"""// Generated by NL2Verilog
// Description: {ltl_formula.description}
// LTL Formula: {ltl_formula.formula}
// Inputs: {', '.join(ltl_formula.inputs)}
// Outputs: {', '.join(ltl_formula.outputs)}

"""
        
        output_file = verilog_file.replace('.v', '_final.v')
        with open(output_file, 'w') as f:
            f.write(header + verilog_content)
        
        return output_file
    
    def generate_testbench(self, verilog_file: str, ltl_formula: LTLFormula, output_dir: str) -> str:
        self.log("Generating testbench...")
        
        module_name = "generated_module"
        tb_file = os.path.join(output_dir, "testbench.v")
        
        clock_signal = next((sig for sig in ltl_formula.inputs if 'clock' in sig.lower() or 'clk' in sig.lower()), None)
        other_inputs = [sig for sig in ltl_formula.inputs if sig != clock_signal]
        
        testbench = f"""// Testbench for generated module
// Description: {ltl_formula.description}

`timescale 1ns / 1ps

module testbench;
    reg clk;
    reg rst;
    """
        
        if other_inputs:
            testbench += "\n    // Inputs\n"
            for sig in other_inputs:
                testbench += f"    reg {sig};\n"
        
        testbench += "\n    // Outputs\n"
        for sig in ltl_formula.outputs:
            testbench += f"    wire {sig};\n"
        
        testbench += f"""
    {module_name} dut ("""
        
        if clock_signal:
            testbench += f"\n        .{clock_signal}(clk),"
        
        for sig in other_inputs:
            testbench += f"\n        .{sig}({sig}),"
        
        for i, sig in enumerate(ltl_formula.outputs):
            testbench += f"\n        .{sig}({sig})"
            if i < len(ltl_formula.outputs) - 1:
                testbench += ","
        
        testbench += "\n    );\n"
        
        testbench += """
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end
    
    initial begin
        $dumpfile("waveform.vcd");
        $dumpvars(0, testbench);
        
        rst = 1;"""
        
        for sig in other_inputs:
            testbench += f"\n        {sig} = 0;"
        
        testbench += """
        
        #20 rst = 0;
        
        #10;
        
        #100 $finish;
    end
    
    initial begin
        $monitor("Time=%0t clk=%b"""
        
        for sig in other_inputs + ltl_formula.outputs:
            testbench += f" {sig}=%b"
        
        testbench += '", $time, clk'
        
        for sig in other_inputs + ltl_formula.outputs:
            testbench += f", {sig}"
        
        testbench += """);
    end
endmodule
"""
        
        with open(tb_file, 'w') as f:
            f.write(testbench)
        
        self.log(f"Generated testbench: {tb_file}")
        return tb_file
    
    def convert(self, specification: str, output_dir: str, keep_intermediate: bool = False) -> Dict[str, str]:
        self.log(f"Starting conversion: {specification}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            ltl_formula = self.nl_to_ltl(specification)
            
            tlsf_file = os.path.join(output_dir, "specification.tlsf")
            self.create_tlsf(ltl_formula, tlsf_file)

            aag_file = self.ltl_to_aiger(ltl_formula, output_dir)
            if not aag_file:
                raise Exception("Failed to convert LTL to AIGER")
            
            verilog_file = self.aiger_to_verilog(aag_file, output_dir)
            if not verilog_file:
                raise Exception("Failed to convert AIGER to Verilog")
            
            final_verilog = self.post_process_verilog(verilog_file, ltl_formula)
            
            testbench_file = self.generate_testbench(final_verilog, ltl_formula, output_dir)
            
            report_file = os.path.join(output_dir, "synthesis_report.txt")
            with open(report_file, 'w') as f:
                f.write(f"Synthesis Report\n")
                f.write(f"================\n\n")
                f.write(f"Specification: {specification}\n")
                f.write(f"LTL Formula: {ltl_formula.formula}\n")
                f.write(f"Inputs: {', '.join(ltl_formula.inputs)}\n")
                f.write(f"Outputs: {', '.join(ltl_formula.outputs)}\n")
                f.write(f"\nGenerated Files:\n")
                f.write(f"- Verilog: {final_verilog}\n")
                f.write(f"- Testbench: {testbench_file}\n")
                f.write(f"- TLSF: {tlsf_file}\n")
            
            if not keep_intermediate:
                for ext in ['.aag', '.aig', '.v']:
                    for file_path_obj in Path(output_dir).glob(f"*{ext}"):
                        if str(file_path_obj) not in [final_verilog, testbench_file]:
                            os.remove(file_path_obj)
            
            self.log("Conversion completed successfully!")
            
            return {
                "verilog": final_verilog,
                "testbench": testbench_file,
                "report": report_file
            }
            
        except Exception as e:
            self.log(f"Conversion failed: {str(e)}", "ERROR")
            raise

@click.command()
@click.argument('specification')
@click.option('--output-dir', '-o', default='./output', help='Output directory')
@click.option('--keep-intermediate', '-k', is_flag=True, help='Keep intermediate files')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--model', '-m', default='gpt-4-turbo-preview', help='OpenAI model to use')
def main(specification: str, output_dir: str, keep_intermediate: bool, verbose: bool, model: str):
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        click.echo("Error: OPENAI_API_KEY not found in environment variables", err=True)
        click.echo("Please set it in a .env file or export it", err=True)
        sys.exit(1)
    
    required_tools = ['ltlsynt', 'aigtoaig', 'abc']
    missing_tools = []
    
    for tool in required_tools:
        if shutil.which(tool) is None:
            missing_tools.append(tool)
    
    if missing_tools:
        click.echo(f"Error: Missing required tools: {', '.join(missing_tools)}", err=True)
        click.echo("Please install them according to the README", err=True)
        sys.exit(1)
    
    converter = NL2Verilog(api_key, model=model, verbose=verbose)
    
    try:
        results = converter.convert(specification, output_dir, keep_intermediate)
        
        click.echo(f"\n✓ Conversion successful!")
        click.echo(f"  Verilog: {results['verilog']}")
        click.echo(f"  Testbench: {results['testbench']}")
        click.echo(f"  Report: {results['report']}")
        
    except Exception as e:
        click.echo(f"\n✗ Conversion failed: {str(e)}", err=True)
        sys.exit(1)

if __name__ == '__main__':
    main()