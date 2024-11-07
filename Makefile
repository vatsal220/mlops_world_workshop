ROOT_DIR := ${PWD}

.PHONY: test-unit
test-unit: # Runs all the unit tests 
	@cd tests && PYTHONPATH=${ROOT_DIR}/src pytest \
		-vv -s $(ROOT_DIR)/tests/unit

.PHONY: test-e2e
test-e2e: # Runs all the e2e tests 
	PYTHONPATH=${ROOT_DIR}/src pytest \
		-vv -s $(ROOT_DIR)/tests/e2e