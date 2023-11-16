//Define all warnings as errors
//List of warning from https://docs.microsoft.com/en-us/windows/win32/direct3dhlsl/hlsl-errors-and-warnings
#pragma warning(error: 3200) //WAR_TYPE_MISMATCH Type mismatches aren't recommended. 
#pragma warning(error: 3201) //WAR_NOFRAGMENTS Fragments aren't recommended. 
#pragma warning(error: 3202) //WAR_INVALID_SEMANTIC The semantic doesn't apply and is ignored. 
#pragma warning(error: 3203) //WAR_SIGNED_UNSIGNED_COMPARE A signed versus unsigned mismatch occurred between destination and value and unsigned is assumed. 
#pragma warning(error: 3204) //WAR_INT_TOO_LARGE Unsigned integer literal is too large so is truncated. 
#pragma warning(error: 3205) //WAR_PRECISION_LOSS In the conversion from larger type to smaller, a loss of data might occur. 
#pragma warning(error: 3206) //WAR_ELT_TRUNCATION The implicit truncation of a vector type occurred. 
#pragma warning(error: 3207) //WAR_CONST_INITIALIZER Initializer was used on a global 'const' variable. This requires setting an external constant. If a literal is wanted, use 'static const' instead. 
#pragma warning(error: 3208) //WAR_FAILED_COMPILING_10L9VS Failed compiling the 10_level_9 (9_x feature levels) vertex shader version of the library function. 
#pragma warning(error: 3209) //WAR_FAILED_COMPILING_10L9PS Failed compiling the 10_level_9 (9_x feature levels) pixel shader version of the library function. 
#pragma warning(error: 3550) //WAR_ARRAY_INDEX_MUST_BE_LITERAL The index of the sampler array must be a literal expression, so the loop is forced to unroll. 
#pragma warning(error: 3551) //WAR_INFINITE_LOOP An infinite loop was detected so the loop writes no values. 
#pragma warning(error: 3552) //WAR_NOT_SIMPLE_LOOP The loop can't be mapped to a shader target because the target doesn't support breaks. 
#pragma warning(error: 3553) //WAR_GRADIENT_WITH_BREAK Can't use gradient instructions in loops with break. 
#pragma warning(error: 3554) //WAR_UNKNOWN_ATTRIBUTE The attribute is unknown or invalid for the specified statement. 
#pragma warning(error: 3555) //WAR_INCOMPATIBLE_FLAGS Flags aren't compatible with the operation. 
#pragma warning(error: 3556) //WAR_INT_DIVIDE_SLOW Integer divides might be much slower, try using uints if possible. 
#pragma warning(error: 3557) //WAR_TOO_SIMPLE_LOOP The loop only executes for a limited number of iterations or doesn't seem to do anything so consider removing it or forcing it to unroll. 
#pragma warning(error: 3558) //WAR_ENDIF_UNINITIALIZED The #endif directive is uninitialized. 
#pragma warning(error: 3559) //WAR_LOOP_ASYMMETRIC_RETURN The loop returns asymmetrically. 
#pragma warning(error: 3560) //WAR_MUST_BRANCH If statements that contain out of bounds array accesses can't be flattened. 
#pragma warning(error: 3561) //WAR_OLDVERSION A particular shader version, such as, ps_1_x, is no longer supported; use the next shader version, such as, ps_2_0. 
#pragma warning(error: 3562) //WAR_OUTOFBOUNDS_LOOPSIM The loop simulation goes out of bounds. 
#pragma warning(error: 3563) //WAR_OUTOFBOUNDS_LOOPUNROLL The loop unrolls out of bounds. 
#pragma warning(error: 3564) //WAR_PRAGMA_RULEDISABLE For better compilation results, consider re-enabling the specified rule. 
#pragma warning(error: 3565) //WAR_DID_NOT_SIMULATE Loop simulation finished early, use /O1 or higher for potentially better codegen. 
#pragma warning(error: 3566) //WAR_NO_EARLY_BREAK Loop won't exit early, try to make sure the loop condition is as tight as possible. 
#pragma warning(error: 3567) //WAR_IGNORING_REGISTER_SEMANTIC The register semantic is ignored. 
#pragma warning(error: 3568) //WAR_UNKNOWN_PRAGMA The unknown pragma directive is ignored. 
#pragma warning(error: 3569) //WAR_LOOP_TOO_LONG The loop executes for more than the maximum number of iterations for the specified shader target, which forces the loop to unroll. 
#pragma warning(error: 3570) //WAR_GRADIENT_MUST_UNROLL A gradient instruction is used in a loop with varying iteration, which forces the loop to unroll. 
#pragma warning(error: 3571) //WAR_POW_NOT_KNOWN_TO_BE_POSITIVE The pow(f, e) intrinsic function won't work for negative f, use abs(f) or conditionally handle negative values if you expect them. 
#pragma warning(error: 3572) //WAR_VARYING_INTERFACE Interface references must resolve to non-varying objects. 
#pragma warning(error: 3573) //WAR_TESSFACTORSCALE_OUTOFRANGE Tessellation factor scale is clamped to the range [0, 1]. 
#pragma warning(error: 3574) //WAR_SYNC_IN_VARYING_FLOW Thread synchronization operations can't be used in varying flow control. 
#pragma warning(error: 3575) //WAR_BREAK_FROM_UAV Automatic unrolling has been disabled for the loop, consider using the [unroll] attribute or manual unrolling. Or, loop termination conditions in varying flow control so can't depend on data read from a UAV. 
#pragma warning(error: 3576) //WAR_OVERRIDDEN_SEMANTIC Patch semantics must live in the enclosed type so the outer semantic is ignored. Or, semantics in type are overridden by variable/function or enclosing type. 
#pragma warning(error: 3577) //WAR_KNOWN_NON_SPECIAL The value can't be infinity, A call to isfinite might not be necessary. /Gis might force isfinite to be performed. Or, The value can't be NaN, A call to isnan might not be necessary. /Gis might force isnan to be performed. 
#pragma warning(error: 3578) //WAR_TLOUT_UNINITIALIZED The output value isn't completely initialized. 
#pragma warning(error: 3579) //WAR_GROUPSHARED_UNSUPPORTED The specified variable doesn't support groupshared so groupshared is ignored. 
#pragma warning(error: 3580) //WAR_CONDITIONAL_SIDE_EFFECT Both sides of the &&, ||, or ?: operator are always evaluated so the side effect on the specified side won't be conditional. 
#pragma warning(error: 3581) //WAR_NO_UNSIGNED_ABS The abs operation on unsigned values is not meaningful so it's ignored. 
#pragma warning(error: 3582) //WAR_TEXTURE_OFFSET Texture access must have literal offset and multisample index. 
#pragma warning(error: 3583) //WAR_POTENTIAL_RACE_CONDITION_UAV A race condition writing to a shared resource was detected, note that threads are writing the same value, but performance might be diminished due to contention. 
#pragma warning(error: 3584) //WAR_POTENTIAL_RACE_CONDITION_GSM A race condition writing to shared memory was detected, note that threads are writing the same value, but performance might be diminished due to contention. 
#pragma warning(error: 3585) //WAR_UNRELIABLE_SOURCE_MARK Source_mark is most useful in /Od builds. Without /Od source_mark, can be moved around in the final shader by optimizations. 
#pragma warning(error: 3586) //WAR_NO_INTERFACE_SUPPORT Abstract interfaces aren't supported on the specified target so interface references must resolve to specific instances. 
#pragma warning(error: 3587) //WAR_MIN10_RCP The target emulates A / B with A * reciprocal(B). If the reciprocal of B is not representable in your min-precision type, the result might not be mathematically correct. 
#pragma warning(error: 3588) //WAR_NO_CLIPPLANES_IN_LIBRARY The clipplanes attribute is ignored in library functions. 
#pragma warning(error: 4008) //WARN_FLOAT_DIVISION_BY_ZERO A floating point division by zero occurred. 
#pragma warning(error: 4114) //WARN_FTOI_OUTOFRANGE The literal floating-point value is out of integer range for the conversion. 
#pragma warning(error: 4115) //WARN_FTOU_OUTOFRANGE The literal floating-point value is out of unsigned integer range for the conversion. 
#pragma warning(error: 4116) //WARN_IDIV_DIVISION_BY_ZERO A possible integer divide by zero occurred. 
#pragma warning(error: 4117) //WARN_UDIV_DIVISION_BY_ZERO A possible unsigned integer divide by zero occurred. 
#pragma warning(error: 4118) //WARN_IMAGINARY_SQUARE_ROOT An imaginary square root operation occurred. 
#pragma warning(error: 4119) //WARN_INDEFINITE_LOG An indefinite logarithm operation occurred. 
#pragma warning(error: 4120) //WARN_REPLACE_NOT_CONVERGE Optimizations aren't converging. 
#pragma warning(error: 4121) //WARN_HOISTING_GRADIENT Gradient-based operations must be moved out of flow control to prevent divergence. Performance might improve by using a non-gradient operation. 
#pragma warning(error: 4122) //WARN_FLOAT_PRECISION_LOSS The sum of two floating point values can't be represented accurately in double precision. 
#pragma warning(error: 4123) //WARN_FLOAT_CLAMP Floating-point operations flush denorm float literals to zero so the specified floating point value is losing precision (this warning will only be shown once per compile). 
#pragma warning(error: 4700) //WAR_GEN_NOT_YET_IMPLEMENTED A feature isn't implemented yet. 
#pragma warning(error: 4701) //WAR_BIAS_MISSED A _bias opportunity was missed because the source wasn't clamped 0 to 1. 
#pragma warning(error: 4702) //WAR_COMP_MISSED A complement opportunity was missed because the input result was clamped from 0 to 1. 
#pragma warning(error: 4703) //WAR_LRP_MISSED Lerp can't be matched because the lerp factor is not _sat'd. 
#pragma warning(error: 4704) //WAR_MAX_CONST_RANGE Literal values outside range -1 to 1 are clamped on all ps_1_x shading models. 
#pragma warning(error: 4705) //WAR_DEPRECATED_INPUT_SEMANTIC The specified input semantic has been deprecated; use the specified semantic instead. 
#pragma warning(error: 4706) //WAR_DEPRECATED_OUTPUT_SEMANTIC The specified output semantic has been deprecated; use the specified semantic instead. 
#pragma warning(error: 4707) //WAR_TEXCOORD_CLAMP The texcoord inputs used directly (that is, other than sampling from textures) in shader body in ps_1_x are always clamped from 0 to 1. 
#pragma warning(error: 4708) //WAR_MIDLEVEL_VARNOTFOUND The mid-level var was not found. 
#pragma warning(error: 4710) //WAR_OLD_SEMANTIC The semantic is no longer in use. 
#pragma warning(error: 4711) //WAR_DUPLICATE_SEMANTIC A duplicate non-system value semantic definition was encountered. 
#pragma warning(error: 4712) //WAR_CANT_MATCH_LOOP The loop can't be matched because the loop count isn't from an integer type. 
#pragma warning(error: 4713) //WAR_BIAS_CLAMPED The sample bias value is limited to the range [-16.00, 15.99] so use the specified value instead of this value. 
#pragma warning(error: 4714) //WAR_CS_TEMP_EXCEEDED The sum of temp registers and indexable temp registers times the specified number of threads exceeds the recommended total number of threads so performance might be reduced. 
#pragma warning(error: 4715) //WAR_UNWRITTEN_SI_VALUE A system-interpreted value is emitted that can't be written in every execution path of the shader. 
#pragma warning(error: 4716) //WAR_PSIZE_HAS_NO_SPECIAL_MEANING The specified semantic has no special meaning on 10_level_9 (9_x feature levels) targets. 
#pragma warning(error: 4717) //WAR_DEPRECATED_FEATURE Effects are deprecated for the D3DCompiler_47.dll or later.