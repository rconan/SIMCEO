function CodeDefine() { 
this.def = new Array();
this.def["rt_OneStep"] = {file: "ert_main_c.html",line:35,type:"fcn"};
this.def["main"] = {file: "ert_main_c.html",line:72,type:"fcn"};
this.def["M1LocalControl_B"] = {file: "M1LocalControl_c.html",line:22,type:"var"};
this.def["M1LocalControl_DW"] = {file: "M1LocalControl_c.html",line:25,type:"var"};
this.def["M1LocalControl_U"] = {file: "M1LocalControl_c.html",line:28,type:"var"};
this.def["M1LocalControl_Y"] = {file: "M1LocalControl_c.html",line:31,type:"var"};
this.def["M1LocalControl_M_"] = {file: "M1LocalControl_c.html",line:34,type:"var"};
this.def["M1LocalControl_M"] = {file: "M1LocalControl_c.html",line:35,type:"var"};
this.def["rate_scheduler"] = {file: "M1LocalControl_c.html",line:43,type:"fcn"};
this.def["M1LocalControl_step"] = {file: "M1LocalControl_c.html",line:56,type:"fcn"};
this.def["M1LocalControl_initialize"] = {file: "M1LocalControl_c.html",line:3811,type:"fcn"};
this.def["M1LocalControl_terminate"] = {file: "M1LocalControl_c.html",line:3836,type:"fcn"};
this.def["B_M1LocalControl_T"] = {file: "M1LocalControl_h.html",line:123,type:"type"};
this.def["DW_M1LocalControl_T"] = {file: "M1LocalControl_h.html",line:211,type:"type"};
this.def["ConstP_M1LocalControl_T"] = {file: "M1LocalControl_h.html",line:346,type:"type"};
this.def["ExtU_M1LocalControl_T"] = {file: "M1LocalControl_h.html",line:351,type:"type"};
this.def["ExtY_M1LocalControl_T"] = {file: "M1LocalControl_h.html",line:356,type:"type"};
this.def["RT_MODEL_M1LocalControl_T"] = {file: "M1LocalControl_types_h.html",line:22,type:"type"};
this.def["M1LocalControl_ConstP"] = {file: "M1LocalControl_data_c.html",line:22,type:"var"};
this.def["int8_T"] = {file: "rtwtypes_h.html",line:49,type:"type"};
this.def["uint8_T"] = {file: "rtwtypes_h.html",line:50,type:"type"};
this.def["int16_T"] = {file: "rtwtypes_h.html",line:51,type:"type"};
this.def["uint16_T"] = {file: "rtwtypes_h.html",line:52,type:"type"};
this.def["int32_T"] = {file: "rtwtypes_h.html",line:53,type:"type"};
this.def["uint32_T"] = {file: "rtwtypes_h.html",line:54,type:"type"};
this.def["int64_T"] = {file: "rtwtypes_h.html",line:55,type:"type"};
this.def["uint64_T"] = {file: "rtwtypes_h.html",line:56,type:"type"};
this.def["real32_T"] = {file: "rtwtypes_h.html",line:57,type:"type"};
this.def["real64_T"] = {file: "rtwtypes_h.html",line:58,type:"type"};
this.def["real_T"] = {file: "rtwtypes_h.html",line:64,type:"type"};
this.def["time_T"] = {file: "rtwtypes_h.html",line:65,type:"type"};
this.def["boolean_T"] = {file: "rtwtypes_h.html",line:66,type:"type"};
this.def["int_T"] = {file: "rtwtypes_h.html",line:67,type:"type"};
this.def["uint_T"] = {file: "rtwtypes_h.html",line:68,type:"type"};
this.def["ulong_T"] = {file: "rtwtypes_h.html",line:69,type:"type"};
this.def["ulonglong_T"] = {file: "rtwtypes_h.html",line:70,type:"type"};
this.def["char_T"] = {file: "rtwtypes_h.html",line:71,type:"type"};
this.def["uchar_T"] = {file: "rtwtypes_h.html",line:72,type:"type"};
this.def["byte_T"] = {file: "rtwtypes_h.html",line:73,type:"type"};
this.def["creal32_T"] = {file: "rtwtypes_h.html",line:83,type:"type"};
this.def["creal64_T"] = {file: "rtwtypes_h.html",line:88,type:"type"};
this.def["creal_T"] = {file: "rtwtypes_h.html",line:93,type:"type"};
this.def["cint8_T"] = {file: "rtwtypes_h.html",line:100,type:"type"};
this.def["cuint8_T"] = {file: "rtwtypes_h.html",line:107,type:"type"};
this.def["cint16_T"] = {file: "rtwtypes_h.html",line:114,type:"type"};
this.def["cuint16_T"] = {file: "rtwtypes_h.html",line:121,type:"type"};
this.def["cint32_T"] = {file: "rtwtypes_h.html",line:128,type:"type"};
this.def["cuint32_T"] = {file: "rtwtypes_h.html",line:135,type:"type"};
this.def["cint64_T"] = {file: "rtwtypes_h.html",line:142,type:"type"};
this.def["cuint64_T"] = {file: "rtwtypes_h.html",line:149,type:"type"};
this.def["pointer_T"] = {file: "rtwtypes_h.html",line:170,type:"type"};
}
CodeDefine.instance = new CodeDefine();
var testHarnessInfo = {OwnerFileName: "", HarnessOwner: "", HarnessName: "", IsTestHarness: "0"};
var relPathToBuildDir = "../ert_main.c";
var fileSep = "/";
var isPC = false;
function Html2SrcLink() {
	this.html2SrcPath = new Array;
	this.html2Root = new Array;
	this.html2SrcPath["ert_main_c.html"] = "../ert_main.c";
	this.html2Root["ert_main_c.html"] = "ert_main_c.html";
	this.html2SrcPath["M1LocalControl_c.html"] = "../M1LocalControl.c";
	this.html2Root["M1LocalControl_c.html"] = "M1LocalControl_c.html";
	this.html2SrcPath["M1LocalControl_h.html"] = "../M1LocalControl.h";
	this.html2Root["M1LocalControl_h.html"] = "M1LocalControl_h.html";
	this.html2SrcPath["M1LocalControl_private_h.html"] = "../M1LocalControl_private.h";
	this.html2Root["M1LocalControl_private_h.html"] = "M1LocalControl_private_h.html";
	this.html2SrcPath["M1LocalControl_types_h.html"] = "../M1LocalControl_types.h";
	this.html2Root["M1LocalControl_types_h.html"] = "M1LocalControl_types_h.html";
	this.html2SrcPath["M1LocalControl_data_c.html"] = "../M1LocalControl_data.c";
	this.html2Root["M1LocalControl_data_c.html"] = "M1LocalControl_data_c.html";
	this.html2SrcPath["rtwtypes_h.html"] = "../rtwtypes.h";
	this.html2Root["rtwtypes_h.html"] = "rtwtypes_h.html";
	this.getLink2Src = function (htmlFileName) {
		 if (this.html2SrcPath[htmlFileName])
			 return this.html2SrcPath[htmlFileName];
		 else
			 return null;
	}
	this.getLinkFromRoot = function (htmlFileName) {
		 if (this.html2Root[htmlFileName])
			 return this.html2Root[htmlFileName];
		 else
			 return null;
	}
}
Html2SrcLink.instance = new Html2SrcLink();
var fileList = [
"ert_main_c.html","M1LocalControl_c.html","M1LocalControl_h.html","M1LocalControl_private_h.html","M1LocalControl_types_h.html","M1LocalControl_data_c.html","rtwtypes_h.html"];
