// This converts 32-bit to 16-bit without changing photon counts 
setMinAndMax(0, 65535); // this defines the 16-bit range
run("16-bit"); // converting normalized to the min and max of 16-bits
run("Enhance Contrast", "saturation=0.35"); // rescale to make it visible. 
//change display settings (these are arbitrary)
Stack.setChannel(1); 
run("Yellow");
setMinAndMax(3,15);
Stack.setChannel(2); 
run("Magenta");
setMinAndMax(3,15);
Stack.setChannel(3); 
run("Cyan");
setMinAndMax(3,50);
Stack.setChannel(4); 
run("Grays");
setMinAndMax(3,50);
