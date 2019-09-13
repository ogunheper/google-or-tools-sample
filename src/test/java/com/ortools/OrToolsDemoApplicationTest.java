package com.ortools;

import com.google.ortools.linearsolver.*;
import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

import static com.google.ortools.linearsolver.MPSolverResponseStatus.MPSOLVER_OPTIMAL;
import static org.junit.jupiter.api.Assertions.assertEquals;

@Slf4j
public class OrToolsDemoApplicationTest {

    @BeforeAll
    public static void beforeAll() {
        log.info("Loading jniortools library...");
        System.loadLibrary("jniortools");
        log.info("Loading jniortools library...Done.");
    }

    private MPConstraintProto getFirstConstraint(double ub) {
        // x + 7y <= ub
        return MPConstraintProto.newBuilder()
                .setLowerBound(Double.NEGATIVE_INFINITY)
                .setUpperBound(ub)
                .addVarIndex(0)
                .addCoefficient(1.0d)
                .addVarIndex(1)
                .addCoefficient(7.0d)
                .build();
    }

    private MPConstraintProto getSecondConstraint(double ub) {
        // x <= ub
        return MPConstraintProto.newBuilder()
                .setLowerBound(Double.NEGATIVE_INFINITY)
                .setUpperBound(ub)
                .addVarIndex(0)
                .addCoefficient(1.0d)
                .build();
    }

    private MPModelProto.Builder getLinearObjective() {
        // Objective: Max. x + 10y
        final MPVariableProto x = this.getIntVar("x").setObjectiveCoefficient(1.0d).build();
        final MPVariableProto y = this.getIntVar("y").setObjectiveCoefficient(10.0d).build();

        return this.getModel(
                Arrays.asList(x, y),
                Arrays.asList(this.getFirstConstraint(17.5d), this.getSecondConstraint(3.5d)),
                Collections.emptyList(),
                Collections.emptyList()
        ).setMaximize(true);
    }

    private MPModelProto.Builder getQuadraticObjective() {
        // Objective: Max. xy
        final MPVariableProto x = this.getIntVar("x").build();
        final MPVariableProto y = this.getIntVar("y").build();

        final MPModelProto.Builder mpModelProto = this.getModel(
                Arrays.asList(x, y),
                Arrays.asList(this.getFirstConstraint(17.5d), this.getSecondConstraint(3.5d)),
                Collections.emptyList(),
                Collections.emptyList()
        ).setMaximize(true);

        final MPQuadraticObjective quadraticObjective = MPQuadraticObjective.newBuilder()
                .addAllQvar1Index(Collections.singletonList(0))
                .addAllQvar2Index(Collections.singletonList(1))
                .addAllCoefficient(Collections.singletonList(2.0d))
                .build();

        mpModelProto
                .setQuadraticObjective(quadraticObjective)
                .setMaximize(true);

        return mpModelProto;
    }

    @Test
    public void shouldSolveLP() {
        final MPModelProto.Builder mpModelProto = this.getLinearObjective();

        final MPSolutionResponse mpSolutionResponse = this.solve(mpModelProto);

        assertEquals(MPSOLVER_OPTIMAL, mpSolutionResponse.getStatus());
        assertEquals(23.0d, mpSolutionResponse.getObjectiveValue());
        assertEquals(3.0d, mpSolutionResponse.getVariableValue(0));
        assertEquals(2.0d, mpSolutionResponse.getVariableValue(1));
    }

    @Test
    public void shouldSolveLPWithQuadraticConstraint() {
        final MPModelProto.Builder mpModelProto = this.getLinearObjective();

        final MPQuadraticConstraint qc0 = MPQuadraticConstraint.newBuilder()
                .addQvar1Index(0)
                .addQvar2Index(1)
                .addQcoefficient(1.0d)
                .setUpperBound(5.0d)
                .build();

        mpModelProto.addGeneralConstraint(MPGeneralConstraintProto.newBuilder().setQuadraticConstraint(qc0).build());

        final MPSolutionResponse mpSolutionResponse = this.solve(mpModelProto);

        assertEquals(MPSOLVER_OPTIMAL, mpSolutionResponse.getStatus());
        assertEquals(22.0d, mpSolutionResponse.getObjectiveValue());
        assertEquals(2.0d, mpSolutionResponse.getVariableValue(0));
        assertEquals(2.0d, mpSolutionResponse.getVariableValue(1));
    }

    @Test
    public void shouldSolveLPWithIndicatorConstraint() {
        final MPModelProto.Builder mpModelProto = this.getLinearObjective();

        final MPVariableProto k = this.getIntVar("k").setUpperBound(1.0d).build();
        mpModelProto
                .addVariable(2, k)
                .removeConstraint(0);

        // x + 7y <= 17.5
        final MPConstraintProto c0_0 = this.getFirstConstraint(17.5d);
        // x + 7y <= 24.5
        final MPConstraintProto c0_1 = this.getFirstConstraint(24.5d);

        final MPIndicatorConstraint ic0 = MPIndicatorConstraint.newBuilder()
                .setVarIndex(2)
                .setVarValue(0)
                .setConstraint(c0_0)
                .build();

        final MPIndicatorConstraint ic1 = MPIndicatorConstraint.newBuilder()
                .setVarIndex(2)
                .setVarValue(1)
                .setConstraint(c0_1)
                .build();

        mpModelProto
                .addGeneralConstraint(MPGeneralConstraintProto.newBuilder().setIndicatorConstraint(ic0).build())
                .addGeneralConstraint(MPGeneralConstraintProto.newBuilder().setIndicatorConstraint(ic1).build());

        final MPSolutionResponse mpSolutionResponse = this.solve(mpModelProto);

        assertEquals(MPSOLVER_OPTIMAL, mpSolutionResponse.getStatus());
        assertEquals(33.0d, mpSolutionResponse.getObjectiveValue());
        assertEquals(3.0d, mpSolutionResponse.getVariableValue(0));
        assertEquals(3.0d, mpSolutionResponse.getVariableValue(1));
        assertEquals(1.0d, mpSolutionResponse.getVariableValue(2));
    }

    @Test
    public void shouldSolveQP() {
        final MPModelProto.Builder mpModelProto = this.getQuadraticObjective();

        final MPSolutionResponse mpSolutionResponse = this.solve(mpModelProto);

        assertEquals(MPSOLVER_OPTIMAL, mpSolutionResponse.getStatus());
        assertEquals(12.0d, mpSolutionResponse.getObjectiveValue());
        assertEquals(3.0d, mpSolutionResponse.getVariableValue(0));
        assertEquals(2.0d, mpSolutionResponse.getVariableValue(1));
    }

    @Test
    public void shouldSolveQPWithQuadraticConstraint() {
        final MPModelProto.Builder mpModelProto = this.getQuadraticObjective();

        final MPQuadraticConstraint qc0 = MPQuadraticConstraint.newBuilder()
                .addQvar1Index(0)
                .addQvar2Index(1)
                .addQcoefficient(1.0d)
                .setUpperBound(5.0d)
                .build();

        mpModelProto.addGeneralConstraint(MPGeneralConstraintProto.newBuilder().setQuadraticConstraint(qc0).build());

        final MPSolutionResponse mpSolutionResponse = this.solve(mpModelProto);

        assertEquals(MPSOLVER_OPTIMAL, mpSolutionResponse.getStatus());
        assertEquals(8.0d, mpSolutionResponse.getObjectiveValue());
        assertEquals(2.0d, mpSolutionResponse.getVariableValue(0));
        assertEquals(2.0d, mpSolutionResponse.getVariableValue(1));
    }

    @Test
    public void shouldSolveQPWithIndicatorConstraint() {
        final MPModelProto.Builder mpModelProto = this.getQuadraticObjective();

        final MPVariableProto k = this.getIntVar("k").setUpperBound(1.0d).build();
        mpModelProto
                .addVariable(2, k)
                .removeConstraint(0);

        // x + 7y <= 17.5
        final MPConstraintProto c0_0 = this.getFirstConstraint(17.5d);
        // x + 7y <= 24.5
        final MPConstraintProto c0_1 = this.getFirstConstraint(24.5d);

        final MPIndicatorConstraint ic0 = MPIndicatorConstraint.newBuilder()
                .setVarIndex(2)
                .setVarValue(0)
                .setConstraint(c0_0)
                .build();

        final MPIndicatorConstraint ic1 = MPIndicatorConstraint.newBuilder()
                .setVarIndex(2)
                .setVarValue(1)
                .setConstraint(c0_1)
                .build();

        mpModelProto
                .addGeneralConstraint(MPGeneralConstraintProto.newBuilder().setIndicatorConstraint(ic0).build())
                .addGeneralConstraint(MPGeneralConstraintProto.newBuilder().setIndicatorConstraint(ic1).build());

        final MPSolutionResponse mpSolutionResponse = this.solve(mpModelProto);

        assertEquals(MPSOLVER_OPTIMAL, mpSolutionResponse.getStatus());
        assertEquals(18.0d, mpSolutionResponse.getObjectiveValue());
        assertEquals(3.0d, mpSolutionResponse.getVariableValue(0));
        assertEquals(3.0d, mpSolutionResponse.getVariableValue(1));
        assertEquals(1.0d, mpSolutionResponse.getVariableValue(2));
    }

    private MPVariableProto.Builder getNumVar(String name) {
        return MPVariableProto.newBuilder()
                .setName(name)
                .setLowerBound(0.0d)
                .setUpperBound(Double.POSITIVE_INFINITY);
    }

    private MPVariableProto.Builder getIntVar(String name) {
        return this.getNumVar(name).setIsInteger(true);
    }

    private MPModelProto.Builder getModel(
            List<MPVariableProto> mpVariableProtos,
            List<MPConstraintProto> mpConstraintProtos,
            List<MPIndicatorConstraint> mpIndicatorConstraints,
            List<MPQuadraticConstraint> mpQuadraticConstraints
    ) {
        final List<MPGeneralConstraintProto> mpIndicatorConstraintProtos = mpIndicatorConstraints.stream()
                .map(MPGeneralConstraintProto.newBuilder()::setIndicatorConstraint)
                .map(MPGeneralConstraintProto.Builder::build)
                .collect(Collectors.toList());

        final List<MPGeneralConstraintProto> mpQuadraticConstraintProtos = mpQuadraticConstraints.stream()
                .map(MPGeneralConstraintProto.newBuilder()::setQuadraticConstraint)
                .map(MPGeneralConstraintProto.Builder::build)
                .collect(Collectors.toList());

        return MPModelProto.newBuilder()
                .addAllVariable(mpVariableProtos)
                .addAllConstraint(mpConstraintProtos)
                .addAllGeneralConstraint(mpIndicatorConstraintProtos)
                .addAllGeneralConstraint(mpQuadraticConstraintProtos);
    }

    private MPSolutionResponse solve(MPModelProto.Builder mpModelProto) {
        final MPModelRequest mpModelRequest = MPModelRequest.newBuilder()
                .setEnableInternalSolverOutput(true)
                .setSolverType(MPModelRequest.SolverType.SCIP_MIXED_INTEGER_PROGRAMMING)
                .setModel(mpModelProto)
                .build();

        return MPSolver.solveWithProto(mpModelRequest);
    }
}
