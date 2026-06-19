import os
import re
import glob

def refactor_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # tep_correction(sigma, sigma_ref, kappa, S)
    # -> tep_correction(sigma, kappa, S, sigma_ref_screened_sq)
    # Wait, in the files, the 4th argument S is sometimes S_vals, S_sample, S_arr, etc.
    # The 2nd argument is sigma_ref or float(sigma_ref)
    
    # Let's match tep_correction(arg1, arg2, arg3, arg4)
    # Be careful not to match too much.
    pattern = re.compile(r'tep_correction\s*\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,)]+)\s*\)')
    
    def repl(m):
        arg1 = m.group(1).strip()
        arg2 = m.group(2).strip() # this was sigma_ref
        arg3 = m.group(3).strip() # this was kappa
        arg4 = m.group(4).strip() # this was S
        
        # We want to change it to: tep_correction(arg1, arg3, arg4, arg2)
        # where arg2 should conceptually become sigma_ref_screened_sq if it was sigma_ref.
        # So we just swap them!
        
        new_arg2 = arg2
        if new_arg2 == "sigma_ref":
            new_arg2 = "sigma_ref_screened_sq"
        elif new_arg2 == "float(sigma_ref)":
            new_arg2 = "float(sigma_ref_screened_sq)"
        elif new_arg2 == "sr":
            # in sensitivity_analysis
            new_arg2 = "sr_sq" # we will need to change sr to sr_sq manually
            
        return f'tep_correction({arg1}, {arg3}, {arg4}, {new_arg2})'

    new_content = pattern.sub(repl, content)
    
    # Special cases:
    new_content = new_content.replace('def optimize_correction(self, df, sigma_ref):', 'def optimize_correction(self, df, sigma_ref_screened_sq):')
    new_content = new_content.replace('def sensitivity_analysis(self, df, fixed_kappa_cep=None):', 'def sensitivity_analysis(self, df, fixed_kappa_cep=None):')
    
    if new_content != content:
        with open(filepath, 'w') as f:
            f.write(new_content)
        print(f"Refactored {filepath}")

for filepath in glob.glob('scripts/steps/*.py'):
    refactor_file(filepath)

